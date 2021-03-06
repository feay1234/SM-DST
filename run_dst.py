# coding=utf-8
#
# Copyright 2020 Heinrich Heine University Duesseldorf
#
# Part of this code is based on the source code of BERT-DST
# (arXiv:1907.03040)
# Part of this code is based on the source code of Transformers
# (arXiv:1910.03771)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import collections
import logging
import os
import random
import glob
import json
import math
import re

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from tensorboardX import SummaryWriter

from transformers import (WEIGHTS_NAME, BertConfig, BertTokenizer)
from transformers import (AdamW, get_linear_schedule_with_warmup)

from SmallCustomBERT import SmallCustomBertForDST
from modeling_bert_dst import (BertForDST)
from data_processors import PROCESSORS
from CustomBERT import CustomBertForDST
from utils_dst import (convert_examples_to_features, DSTExample, convert_to_unicode, InputFeatures)
from tensorlistdataset import (TensorListDataset)

logger = logging.getLogger(__name__)

ALL_MODELS = tuple(BertConfig.pretrained_config_archive_map.keys())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForDST, BertTokenizer),
    'custombert': (BertConfig, CustomBertForDST, BertTokenizer),
    'smallcustombert': (BertConfig, SmallCustomBertForDST, BertTokenizer),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def batch_to_device(batch, device):
    batch_on_device = []
    for element in batch:
        if isinstance(element, dict):
            batch_on_device.append({k: v.to(device) for k, v in element.items()})
        else:
            batch_on_device.append(element.to(device))
    return tuple(batch_on_device)


def train_meta(args, train_dataset, features, model, tokenizer, processor, continue_from_global_step=0):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    if args.save_epochs > 0:
        args.save_steps = t_total // args.num_train_epochs * args.save_epochs

    num_warmup_steps = int(t_total * args.warmup_proportion)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    model_single_gpu = model
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model_single_gpu)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running meta training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Warmup steps = %d", num_warmup_steps)

    if continue_from_global_step > 0:
        logger.info("Fast forwarding to global step %d to resume training from latest checkpoint...",
                    continue_from_global_step)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    max_turn = 50

    for _ in train_iterator:

        epoch_iterator = tqdm(train_dataloader, desc="Meta Iteration", disable=args.local_rank not in [-1, 0])

        for step, batch in enumerate(epoch_iterator):
            # If training is continued from a checkpoint, fast forward
            # to the state of that checkpoint.
            if global_step < continue_from_global_step:
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    scheduler.step()  # Update learning rate schedule
                    global_step += 1
                continue

            model.train()
            batch = batch_to_device(batch, args.device)
            inputs = {'input_ids': batch[0],
                      'class_label_id': batch[1],
                      'trainMeta': True}

            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # Log metrics
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

        if args.local_rank in [-1, 0]:
            tb_writer.close()

        return global_step, tr_loss / global_step


def train(args, train_dataset, features, model, tokenizer, processor, continue_from_global_step=0):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    # feay1234
    # Initialisation
    dialog_ids = list(set([f.did for f in features]))
    dialog2batch = collections.defaultdict(list)
    for batch, raw in zip(train_dataset, features):
        dialog2batch[raw.did].append([batch, raw])

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    if args.save_epochs > 0:
        args.save_steps = t_total // args.num_train_epochs * args.save_epochs

    num_warmup_steps = int(t_total * args.warmup_proportion)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    model_single_gpu = model
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model_single_gpu)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Warmup steps = %d", num_warmup_steps)

    if continue_from_global_step > 0:
        logger.info("Fast forwarding to global step %d to resume training from latest checkpoint...",
                    continue_from_global_step)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    max_turn = 50

    for _ in train_iterator:

        # epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])

        rand_dialogs = [None] * args.per_gpu_train_batch_size
        finish_dialogs = set()
        dialog_pool = dialog_ids.copy()
        random.shuffle(dialog_pool)
        memo = {}
        pbar = tqdm(total=len(dialog_ids), desc="Iteration", disable=args.local_rank not in [-1, 0])
        step = 0

        # cls = torch.tensor([[[1] * model.hidden_size] * (max_turn-1)] * args.per_gpu_train_batch_size, dtype=torch.float)
        # mask_cls = torch.tensor([[0] * max_turn] * args.per_gpu_train_batch_size, dtype=torch.float)
        # token_type_cls = torch.tensor([[0] * max_turn] * args.per_gpu_train_batch_size, dtype=torch.long)

        while True:

            if len(finish_dialogs) == len(dialog_ids):
                pbar.close()
                break

            for i in range(args.per_gpu_train_batch_size):
                if rand_dialogs[i] is not None:
                    continue

                # if run out of candidate dialog, reset the pool
                if len(dialog_pool) == 0:
                    dialog_pool = dialog_ids.copy()
                    random.shuffle(dialog_pool)

                rand_dialogs[i] = dialog_pool.pop()

            batch = []
            for i in range(args.per_gpu_train_batch_size):
                dialog = rand_dialogs[i]
                if dialog not in memo:
                    memo[dialog] = 0

                # turn = memo[dialog]
                # if turn == 0:
                #     cls[i] = torch.tensor([[1.0] * model.hidden_size] * (max_turn-1), dtype=torch.float)
                #     mask_cls[i] = torch.tensor([1.0] + [0.0] * (max_turn-1))
                #     token_type_cls[i] = torch.tensor([0.0] * max_turn)
                # else:
                #     cls[i][turn - 1] = prev_cls[i]
                #     mask_cls[i][:turn+1] = 1
                #     token_type_cls[i][1:turn+1] = 1
                # sanity check
                # prev_cls[i] = torch.tensor([1] * model.hidden_size)

                batch.append(dialog2batch[dialog][memo[dialog]][0])

                memo[dialog] += 1
                if memo[dialog] >= len(dialog2batch[dialog]):
                    del memo[dialog]
                    finish_dialogs.add(dialog)
                    rand_dialogs[i] = None
                    pbar.update(1)
            # print("old")
            # for step, batch in enumerate(epoch_iterator):
            #     print(batch)
            #     break
            stack_batch = []
            # print(len(batch[0]))
            for idx in range(len(batch[0])):
                if idx in [0, 1, 2, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]:
                    stack_batch.append(torch.stack([b[idx] for b in batch], dim=0))
                else:
                    tmp = {}
                    for key in batch[0][idx]:
                        tmp[key] = torch.tensor([b[idx][key].item() for b in batch], dtype=torch.long)
                    stack_batch.append(tmp)

            batch = stack_batch

            # for step, batch in enumerate(epoch_iterator):
            #     print(step)
            # If training is continued from a checkpoint, fast forward
            # to the state of that checkpoint.
            if global_step < continue_from_global_step:
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    scheduler.step()  # Update learning rate schedule
                    global_step += 1
                continue

            model.train()

            batch = batch_to_device(batch, args.device)

            # print(len(batch))
            # for i in range(len(batch)):
            #     print(i, batch[i])
            # print(batch)

            # This is what is forwarded to the "forward" def.
            if len(batch) == 11:
                inputs = {'input_ids': batch[0],
                          'input_mask': batch[1],
                          'segment_ids': batch[2],
                          'start_pos': batch[3],
                          'end_pos': batch[4],
                          'inform_slot_id': batch[5],
                          'refer_id': batch[6],
                          'diag_state': batch[7],
                          'class_label_id': batch[8],
                          'turn_weight': batch[-1],
                          'set_type': "train",
                          'target': args.target,
                          'fewshot': args.fewshot}
            if len(batch) == 17:
                inputs = {'input_ids': batch[0],
                          'input_mask': batch[1],
                          'segment_ids': batch[2],
                          'start_pos': batch[3],
                          'end_pos': batch[4],
                          'inform_slot_id': batch[5],
                          'refer_id': batch[6],
                          'diag_state': batch[7],
                          'class_label_id': batch[8],
                          'refer_tokens': batch[-7],
                          'refer_marked': batch[-6],
                          'refer_segment': batch[-5],
                          'inform_tokens': batch[-4],
                          'inform_marked': batch[-3],
                          'inform_segment': batch[-2],
                          'turn_weight': batch[-1],
                          'set_type': "train",
                          'target': args.target,
                          'fewshot': args.fewshot}
            elif len(batch) == 14:
                inputs = {'input_ids': batch[0],
                          'input_mask': batch[1],
                          'segment_ids': batch[2],
                          'start_pos': batch[3],
                          'end_pos': batch[4],
                          'inform_slot_id': batch[5],
                          'refer_id': batch[6],
                          'diag_state': batch[7],
                          'class_label_id': batch[8],
                          'refer_tokens': batch[-4],
                          'inform_tokens': batch[-3],
                          'turn_ids': batch[-2],
                          'turn_weight': batch[-1],
                          'set_type': "train",
                          'target': args.target,
                          'fewshot': args.fewshot}

            if args.model_type == "smallcustombert":
                for slot in model.slot_list:
                    inputs['slot'] = slot
                    outputs = model(**inputs)
                    loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

                    if args.n_gpu > 1:
                        loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps

                    if args.fp16:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    tr_loss += loss.item()

            else:
                outputs = model(**inputs)
                loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # Log metrics
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                # Save model checkpoint
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model,
                                                            'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                # epoch_iterator.close()
                break

            step += 1
        if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
            results = evaluate(args, model_single_gpu, tokenizer, processor, prefix=global_step)
            for key, value in results.items():
                tb_writer.add_scalar('eval_{}'.format(key), value, global_step)

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, processor, prefix=""):
    dataset, features = load_and_cache_examples(args, model, tokenizer, processor, evaluate=True)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size
    eval_sampler = SequentialSampler(dataset)  # Note that DistributedSampler samples randomly
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_results = []
    all_preds = []
    ds = {slot: 'none' for slot in model.slot_list}
    with torch.no_grad():
        diag_state = {slot: torch.tensor([0 for _ in range(args.eval_batch_size)]).to(args.device) for slot in
                      model.slot_list}

    # last_cls = torch.tensor([1.0] * model.hidden_size, dtype=torch.float)
    max_turn = 50
    cls = torch.tensor([[[1] * model.hidden_size] * (max_turn - 1)] * args.eval_batch_size, dtype=torch.float)
    mask_cls = torch.tensor([[0] * max_turn] * args.eval_batch_size, dtype=torch.float)
    token_type_cls = torch.tensor([[0] * max_turn] * args.per_gpu_train_batch_size, dtype=torch.long)

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = batch_to_device(batch, args.device)

        # Reset dialog state if turn is first in the dialog.
        turn_itrs = [features[i.item()].guid.split('-')[2] for i in batch[9]]
        reset_diag_state = np.where(np.array(turn_itrs) == '0')[0]
        for slot in model.slot_list:
            for i in reset_diag_state:
                diag_state[slot][i] = 0

        with torch.no_grad():

            if len(batch) == 11:
                inputs = {'input_ids': batch[0],
                          'input_mask': batch[1],
                          'segment_ids': batch[2],
                          'start_pos': batch[3],
                          'end_pos': batch[4],
                          'inform_slot_id': batch[5],
                          'refer_id': batch[6],
                          'diag_state': diag_state,
                          'class_label_id': batch[8],
                          'turn_weight': batch[-1],
                          'set_type': args.predict_type,
                          'target': args.target}
            elif len(batch) == 12:
                inputs = {'input_ids': batch[0],
                          'input_mask': batch[1],
                          'segment_ids': batch[2],
                          'start_pos': batch[3],
                          'end_pos': batch[4],
                          'inform_slot_id': batch[5],
                          'refer_id': batch[6],
                          'diag_state': diag_state,
                          'class_label_id': batch[8],
                          'turn_ids': batch[-2],
                          'turn_weight': batch[-1],
                          'set_type': args.predict_type,
                          'target': args.target}
            elif len(batch) == 17:
                inputs = {'input_ids': batch[0],
                          'input_mask': batch[1],
                          'segment_ids': batch[2],
                          'start_pos': batch[3],
                          'end_pos': batch[4],
                          'inform_slot_id': batch[5],
                          'refer_id': batch[6],
                          'diag_state': batch[7],
                          'class_label_id': batch[8],
                          'refer_tokens': batch[-7],
                          'refer_marked': batch[-6],
                          'refer_segment': batch[-5],
                          'inform_tokens': batch[-4],
                          'inform_marked': batch[-3],
                          'inform_segment': batch[-2],
                          'turn_weight': batch[-1],
                          'set_type': args.predict_type,
                          'target': args.target}

            unique_ids = [features[i.item()].guid for i in batch[9]]
            values = [features[i.item()].values for i in batch[9]]
            input_ids_unmasked = [features[i.item()].input_ids_unmasked for i in batch[9]]
            inform = [features[i.item()].inform for i in batch[9]]

            if args.model_type == "smallcustombert":
                total_loss = 0
                per_slot_per_example_loss = {}
                per_slot_class_logits = {}
                per_slot_start_logits = {}
                per_slot_end_logits = {}
                per_slot_refer_logits = {}
                for slot in model.slot_list:
                    inputs['slot'] = slot
                    outputs = model(**inputs)
                    total_loss += outputs[0]
                    per_slot_per_example_loss[slot] = outputs[1]
                    per_slot_class_logits[slot] = outputs[2]
                    per_slot_start_logits[slot] = outputs[3]
                    per_slot_end_logits[slot] = outputs[4]
                    per_slot_refer_logits[slot] = outputs[5]

                    desc_tokens = model.tokenizer.convert_tokens_to_ids(
                        model.tokenizer.tokenize(model.desc[slot].split(" [SEP]")[0]))

                    # input_ids = torch.cat((torch.cat(
                    #     (inputs['input_ids'][:, 0:1], torch.tensor([desc_tokens] * inputs['input_ids'].shape[0]).to(args.device)), 1),
                    #                        inputs['input_ids'][:, 1:]), 1).to(args.device)

                    # inputs['start_pos'][slot] = torch.mul(inputs['start_pos'][slot] + len(desc_tokens), (inputs['start_pos'][slot] > 0).long())
                    # inputs['end_pos'][slot] = torch.mul(inputs['end_pos'][slot] + len(desc_tokens), (inputs['end_pos'][slot] > 0).long())

                    # print(slot)
                    # if inputs['start_pos'][slot].cpu().numpy().sum() != 0:
                    #     for i, j in zip(input_ids.cpu().numpy(), inputs['start_pos'][slot].cpu().numpy()):
                    #         if i[j] == 101:
                    #             continue
                    #         print(slot, model.tokenizer.convert_ids_to_tokens(torch.tensor([i[j]])))
                    # print(input_ids.size())
                    # print(outputs[3].size())
                    # print("----------")

                outputs = (total_loss,) + (
                    per_slot_per_example_loss, per_slot_class_logits, per_slot_start_logits, per_slot_end_logits,
                    per_slot_refer_logits,)

            else:
                outputs = model(**inputs)

            # Update dialog state for next turn.
            for slot in model.slot_list:
                if inputs['set_type'] == "test" and inputs['target'] != "" and inputs['target'] not in slot:
                    continue
                elif inputs['set_type'] != "test" and inputs['target'] != "" and inputs['target'] in slot:
                    continue

                updates = outputs[2][slot].max(1)[1]
                for i, u in enumerate(updates):
                    if u != 0:
                        diag_state[slot][i] = u

        results = eval_metric(model, inputs, outputs[0], outputs[1], outputs[2], outputs[3], outputs[4], outputs[5])
        preds, ds = predict_and_format(model, tokenizer, inputs, outputs[2], outputs[3], outputs[4], outputs[5],
                                       unique_ids, input_ids_unmasked, values, inform, prefix, ds)
        all_results.append(results)
        all_preds.append(preds)

    all_preds = [item for sublist in all_preds for item in sublist]  # Flatten list

    # Generate final results
    final_results = {}
    for k in all_results[0].keys():
        final_results[k] = torch.stack([r[k] for r in all_results]).mean()

    # Write final predictions (for evaluation with external tool)
    output_prediction_file = os.path.join(args.output_dir, "pred_res.%s.%s.json" % (args.predict_type, prefix))
    with open(output_prediction_file, "w") as f:
        json.dump(all_preds, f, indent=2)

    return final_results


def eval_metric(model, features, total_loss, per_slot_per_example_loss, per_slot_class_logits, per_slot_start_logits,
                per_slot_end_logits, per_slot_refer_logits):
    metric_dict = {}
    per_slot_correctness = {}
    for slot in model.slot_list:

        if features['set_type'] == "test" and features['target'] != "" and features['target'] not in slot:
            continue
        elif features['set_type'] != "test" and features['target'] != "" and features['target'] in slot:
            continue

        per_example_loss = per_slot_per_example_loss[slot]
        class_logits = per_slot_class_logits[slot]
        start_logits = per_slot_start_logits[slot]
        end_logits = per_slot_end_logits[slot]
        refer_logits = per_slot_refer_logits[slot]

        class_label_id = features['class_label_id'][slot]
        start_pos = features['start_pos'][slot]
        end_pos = features['end_pos'][slot]
        refer_id = features['refer_id'][slot]

        _, class_prediction = class_logits.max(1)
        class_correctness = torch.eq(class_prediction, class_label_id).float()
        class_accuracy = class_correctness.mean()

        # "is pointable" means whether class label is "copy_value",
        # i.e., that there is a span to be detected.
        token_is_pointable = torch.eq(class_label_id, model.class_types.index('copy_value')).float()
        _, start_prediction = start_logits.max(1)
        start_correctness = torch.eq(start_prediction, start_pos).float()
        _, end_prediction = end_logits.max(1)
        end_correctness = torch.eq(end_prediction, end_pos).float()
        token_correctness = start_correctness * end_correctness
        token_accuracy = (token_correctness * token_is_pointable).sum() / token_is_pointable.sum()
        # NaNs mean that none of the examples in this batch contain spans. -> division by 0
        # The accuracy therefore is 1 by default. -> replace NaNs
        if math.isnan(token_accuracy):
            token_accuracy = torch.tensor(1.0, device=token_accuracy.device)

        token_is_referrable = torch.eq(class_label_id,
                                       model.class_types.index('refer') if 'refer' in model.class_types else -1).float()
        _, refer_prediction = refer_logits.max(1)
        refer_correctness = torch.eq(refer_prediction,
                                     refer_id if model.config.version not in [6, 7] else refer_id > 0).float()
        refer_accuracy = refer_correctness.sum() / token_is_referrable.sum()
        # NaNs mean that none of the examples in this batch contain referrals. -> division by 0
        # The accuracy therefore is 1 by default. -> replace NaNs
        if math.isnan(refer_accuracy) or math.isinf(refer_accuracy):
            refer_accuracy = torch.tensor(1.0, device=refer_accuracy.device)

        total_correctness = class_correctness * (token_is_pointable * token_correctness + (1 - token_is_pointable)) * (
                token_is_referrable * refer_correctness + (1 - token_is_referrable))
        total_accuracy = total_correctness.mean()

        loss = per_example_loss.mean()
        metric_dict['eval_accuracy_class_%s' % slot] = class_accuracy
        metric_dict['eval_accuracy_token_%s' % slot] = token_accuracy
        metric_dict['eval_accuracy_refer_%s' % slot] = refer_accuracy
        metric_dict['eval_accuracy_%s' % slot] = total_accuracy
        metric_dict['eval_loss_%s' % slot] = loss
        per_slot_correctness[slot] = total_correctness

    goal_correctness = torch.stack([c for c in per_slot_correctness.values()], 1).prod(1)
    goal_accuracy = goal_correctness.mean()
    metric_dict['eval_accuracy_goal'] = goal_accuracy
    metric_dict['loss'] = total_loss
    return metric_dict


def predict_and_format(model, tokenizer, features, per_slot_class_logits, per_slot_start_logits, per_slot_end_logits,
                       per_slot_refer_logits, ids, input_ids_unmasked, values, inform, prefix, ds):
    prediction_list = []
    dialog_state = ds
    # print(ids)
    for i in range(len(ids)):
        if int(ids[i].split("-")[2]) == 0:
            dialog_state = {slot: 'none' for slot in model.slot_list}

        prediction = {}
        prediction_addendum = {}
        for slot in model.slot_list:

            if features['set_type'] == "test" and features['target'] != "" and features['target'] not in slot:
                continue
            elif features['set_type'] != "test" and features['target'] != "" and features['target'] in slot:
                continue

            class_logits = per_slot_class_logits[slot][i]
            start_logits = per_slot_start_logits[slot][i]
            end_logits = per_slot_end_logits[slot][i]
            refer_logits = per_slot_refer_logits[slot][i]

            input_ids = features['input_ids'][i].tolist()
            class_label_id = int(features['class_label_id'][slot][i])
            start_pos = int(features['start_pos'][slot][i])
            end_pos = int(features['end_pos'][slot][i])
            if model.config.version in [6, 7]:
                refer_id = 0 if int(features['refer_id'][slot][i]) < 1 else 1
            else:
                refer_id = int(features['refer_id'][slot][i])

            class_prediction = int(class_logits.argmax())
            start_prediction = int(start_logits.argmax())
            end_prediction = int(end_logits.argmax())
            refer_prediction = int(refer_logits.argmax())

            prediction['guid'] = ids[i].split("-")
            prediction['class_prediction_%s' % slot] = class_prediction
            prediction['class_label_id_%s' % slot] = class_label_id
            prediction['start_prediction_%s' % slot] = start_prediction
            prediction['start_pos_%s' % slot] = start_pos
            prediction['end_prediction_%s' % slot] = end_prediction
            prediction['end_pos_%s' % slot] = end_pos
            prediction['refer_prediction_%s' % slot] = refer_prediction
            prediction['refer_id_%s' % slot] = refer_id
            prediction['input_ids_%s' % slot] = input_ids

            if class_prediction == model.class_types.index('dontcare'):
                dialog_state[slot] = 'dontcare'
            elif class_prediction == model.class_types.index('copy_value'):
                input_tokens = tokenizer.convert_ids_to_tokens(input_ids_unmasked[i])
                dialog_state[slot] = ' '.join(input_tokens[start_prediction:end_prediction + 1])
                dialog_state[slot] = re.sub("(^| )##", "", dialog_state[slot])
            elif 'true' in model.class_types and class_prediction == model.class_types.index('true'):
                dialog_state[slot] = 'true'
            elif 'false' in model.class_types and class_prediction == model.class_types.index('false'):
                dialog_state[slot] = 'false'
            elif class_prediction == model.class_types.index('inform'):
                dialog_state[slot] = '????' + inform[i][slot]
            # Referral case is handled below

            prediction_addendum['slot_prediction_%s' % slot] = dialog_state[slot]
            prediction_addendum['slot_groundtruth_%s' % slot] = values[i][slot]

        # Referral case. All other slot values need to be seen first in order
        # to be able to do this correctly.
        for slot in model.slot_list:

            if features['set_type'] == "test" and features['target'] != "" and features['target'] not in slot:
                continue
            elif features['set_type'] != "test" and features['target'] != "" and features['target'] in slot:
                continue

            class_logits = per_slot_class_logits[slot][i]
            refer_logits = per_slot_refer_logits[slot][i]

            class_prediction = int(class_logits.argmax())
            refer_prediction = int(refer_logits.argmax())

            if 'refer' in model.class_types and class_prediction == model.class_types.index('refer'):
                # Only slots that have been mentioned before can be referred to.
                # One can think of a situation where one slot is referred to in the same utterance.
                # This phenomenon is however currently not properly covered in the training data
                # label generation process.
                if model.config.version in [6, 7]:
                    dialog_state[slot] = dialog_state[slot] if refer_prediction == 1 else ""
                else:
                    dialog_state[slot] = dialog_state[model.slot_list[refer_prediction - 1]]

                prediction_addendum['slot_prediction_%s' % slot] = dialog_state[slot]  # Value update

        prediction.update(prediction_addendum)
        prediction_list.append(prediction)

    return prediction_list, dialog_state


def load_and_cache_examples(args, model, tokenizer, processor, evaluate=False, index=-1):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Load data features from cache or dataset file
    cached_file = args.output_dir + "/" + 'cached_{}_features'.format(
        args.predict_type if evaluate else 'train')

    if os.path.exists(cached_file) and not args.overwrite_cache:  # and not output_examples:
        logger.info("Loading features from cached file %s", cached_file)
        features = torch.load(cached_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        processor_args = {'append_history': args.append_history,
                          'use_history_labels': args.use_history_labels,
                          'swap_utterances': args.swap_utterances,
                          'label_value_repetitions': args.label_value_repetitions,
                          'delexicalize_sys_utts': args.delexicalize_sys_utts,
                          'local_machine': args.local_machine,
                          'seq_num': args.seq_num,
                          'perturbation': args.perturbation,
                          'value_augment': args.value_augment,
                          'enable_turn_ids': args.enable_turn_ids,
                          'source': args.source,
                          'target': args.target,
                          'enable_ds': args.enable_ds,
                          'fewshot': args.fewshot,
                          'index': index,
                          'batch_size': args.per_gpu_train_batch_size}

        if evaluate and args.predict_type == "dev":
            examples = processor.get_dev_examples(args.data_dir, processor_args)
        elif evaluate and args.predict_type == "test":
            examples = processor.get_test_examples(args.data_dir, processor_args)
        else:
            examples = processor.get_train_examples(args.data_dir, processor_args)

        # add turn weight
        guid2maxturn = collections.defaultdict(int)
        for e in examples:
            guid2maxturn[e.did] = max(guid2maxturn[e.did], e.turn)
        for e in examples:
            e.turn_weight = e.turn_weight + ((guid2maxturn[e.did] - e.turn) * args.turn_weight)
            e.max_turn = guid2maxturn[e.did]

        # print(model.slot_list)
        features = convert_examples_to_features(examples=examples,
                                                slot_list=model.slot_list,
                                                class_types=model.class_types,
                                                model_type=args.model_type,
                                                tokenizer=tokenizer,
                                                max_seq_length=args.max_seq_length,
                                                slot_value_dropout=(0.0 if evaluate else args.svd))
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_file)
            # feay1234
            if not args.local_machine and index == -1:
                torch.save(features, cached_file)
            # torch.save(features, cached_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    f_refer_ids = [f.refer_id for f in features]
    f_start_pos = [f.start_pos for f in features]
    f_end_pos = [f.end_pos for f in features]
    f_inform_slot_ids = [f.inform_slot for f in features]
    f_diag_state = [f.diag_state for f in features]
    f_class_label_ids = [f.class_label_id for f in features]



    all_start_positions = {}
    all_end_positions = {}
    all_inform_slot_ids = {}
    all_diag_state = {}
    all_class_label_ids = {}
    all_refer_ids = {}

    for s in model.slot_list:
        all_start_positions[s] = torch.tensor([f[s] for f in f_start_pos], dtype=torch.long)
        all_end_positions[s] = torch.tensor([f[s] for f in f_end_pos], dtype=torch.long)
        all_inform_slot_ids[s] = torch.tensor([f[s] for f in f_inform_slot_ids], dtype=torch.long)
        all_refer_ids[s] = torch.tensor([f[s] for f in f_refer_ids], dtype=torch.long)
        all_diag_state[s] = torch.tensor([f[s] for f in f_diag_state], dtype=torch.long)
        all_class_label_ids[s] = torch.tensor([f[s] for f in f_class_label_ids], dtype=torch.long)

    # feay1234
    all_turn_weight = torch.tensor([f.turn_weight for f in features])


    dataset = TensorListDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_start_positions, all_end_positions,
                                all_inform_slot_ids,
                                all_refer_ids,
                                all_diag_state,
                                all_class_label_ids, all_example_index,
                                all_turn_weight)

    return dataset, features


def load_and_cache_meta_examples(args, model, tokenizer, processor, evaluate=False):
    path = "data/dialogues/"
    files = os.listdir(path)
    meta = []

    for f in files:
        with open(path + f, "r", encoding='utf-8') as reader:
            for line in reader:
                meta.append(json.loads(line))

    vehicle_labels = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    reserve_labels = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    select = {"BUS_SCHEDULE_BOT": vehicle_labels,
              "EVENT_RESERVE": reserve_labels,
              "RESTAURANT_PICKER": reserve_labels,
              "APARTMENT_FINDER": reserve_labels,
              "MAKE_RESTAURANT_RESERVATIONS": reserve_labels}

    features = []
    for i in range(len(meta)):

        try:
            domain = meta[i]['domain']
        except:
            continue
        if domain not in select:
            continue
        text = " ".join(meta[i]['turns'][1:])
        text = convert_to_unicode(text)
        input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("[CLS] " + text + " [SEP]"))
        while len(input_ids) < args.max_seq_length:
            input_ids.append(0)
        input_ids = input_ids[:args.max_seq_length]
        class_label_id = {cls: label for cls, label in zip(model.slot_list, select[domain])}
        features.append(InputFeatures(input_ids, class_label_id=class_label_id))

    # if args.local_rank in [-1, 0]:
    #     logger.info("Saving features into cached file %s", cached_file)
    #     feay1234
    # if not args.local_machine:
    #     torch.save(features, cached_file)
    # torch.save(features, cached_file)
    #
    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    f_class_label_ids = [f.class_label_id for f in features]

    all_class_label_ids = {}
    print(len(f_class_label_ids), len(model.slot_list))

    # print(model.slot_list)
    for s in model.slot_list:
        all_class_label_ids[s] = torch.tensor([f[s] for f in f_class_label_ids], dtype=torch.long)

    dataset = TensorListDataset(all_input_ids, all_class_label_ids)

    return dataset, features


def read_dataset_config(file):
    with open(file, "r", encoding='utf-8') as reader:
        data = json.load(reader)
        return data


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="Name of the task (e.g., multiwoz21).")
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="Task database.")
    parser.add_argument("--dataset_config", default=None, type=str, required=True,
                        help="Dataset configuration file.")
    parser.add_argument("--predict_type", default=None, type=str, required=True,
                        help="Portion of the data to perform prediction on (e.g., dev, test).")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(
                            ALL_MODELS))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")

    # feay1234
    parser.add_argument("--mode", default="", type=str)
    parser.add_argument("--turn_weight", default=0.1, type=float,
                        help="enable turn weight")
    parser.add_argument('--local_machine', default=False, type=bool,
                        help="local machine")
    parser.add_argument('--seq_num', default=0, type=int)
    parser.add_argument('--perturbation', default=0, type=int)
    parser.add_argument('--value_augment', default=0, type=int)
    parser.add_argument('--enable_turn_ids', default=0, type=int)

    parser.add_argument('--source', default="", type=str)
    parser.add_argument('--target', default="", type=str)
    parser.add_argument('--fewshot', default=0, type=int)

    parser.add_argument('--option', default="col-desc", type=str)
    parser.add_argument('--enable_ds', default=0, type=int, help="Enable distant supervision")
    parser.add_argument('--version', default=6, type=int, help="version of efficient model")
    parser.add_argument('--max_slot_len', default=32, type=int)
    parser.add_argument('--enableInterEmb', default=0, type=int)

    # Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")

    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="Maximum input length after tokenization. Longer sequences will be truncated, shorter ones padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the <predict_type> set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--dropout_rate", default=0.3, type=float,
                        help="Dropout rate for BERT representations.")
    parser.add_argument("--heads_dropout", default=0.0, type=float,
                        help="Dropout rate for classification heads.")
    parser.add_argument("--class_loss_ratio", default=0.8, type=float,
                        help="The ratio applied on class loss in total loss calculation. "
                             "Should be a value in [0.0, 1.0]. "
                             "The ratio applied on token loss is (1-class_loss_ratio)/2. "
                             "The ratio applied on refer loss is (1-class_loss_ratio)/2.")
    parser.add_argument("--token_loss_for_nonpointable", action='store_true',
                        help="Whether the token loss for classes other than copy_value contribute towards total loss.")
    parser.add_argument("--refer_loss_for_nonpointable", action='store_true',
                        help="Whether the refer loss for classes other than refer contribute towards total loss.")

    parser.add_argument("--append_history", action='store_true',
                        help="Whether or not to append the dialog history to each turn.")
    parser.add_argument("--use_history_labels", action='store_true',
                        help="Whether or not to label the history as well.")
    parser.add_argument("--swap_utterances", action='store_true',
                        help="Whether or not to swap the turn utterances (default: sys|usr, swapped: usr|sys).")
    parser.add_argument("--label_value_repetitions", action='store_true',
                        help="Whether or not to label values that have been mentioned before.")
    parser.add_argument("--delexicalize_sys_utts", action='store_true',
                        help="Whether or not to delexicalize the system utterances.")
    parser.add_argument("--class_aux_feats_inform", action='store_true',
                        help="Whether or not to use the identity of informed slots as auxiliary featurs for class prediction.")
    parser.add_argument("--class_aux_feats_ds", action='store_true',
                        help="Whether or not to use the identity of slots in the current dialog state as auxiliary featurs for class prediction.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_proportion", default=0.0, type=float,
                        help="Linear warmup over warmup_proportion * steps.")
    parser.add_argument("--svd", default=0.0, type=float,
                        help="Slot value dropout ratio (default: 0.0)")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=0,
                        help="Save checkpoint every X updates steps. Overwritten by --save_epochs.")
    parser.add_argument('--save_epochs', type=int, default=0,
                        help="Save checkpoint every X epochs. Overrides --save_steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    args = parser.parse_args()

    assert (args.warmup_proportion >= 0.0 and args.warmup_proportion <= 1.0)
    assert (args.svd >= 0.0 and args.svd <= 1.0)
    assert (args.class_aux_feats_ds is False or args.per_gpu_eval_batch_size == 1)
    assert (not args.class_aux_feats_inform or args.per_gpu_eval_batch_size == 1)
    assert (not args.class_aux_feats_ds or args.per_gpu_eval_batch_size == 1)

    task_name = args.task_name.lower()
    if task_name not in PROCESSORS:
        raise ValueError("Task not found: %s" % (task_name))

    processor = PROCESSORS[task_name](args.dataset_config)
    dst_slot_list = processor.slot_list
    dst_class_types = processor.class_types
    dst_class_labels = len(dst_class_types)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        # args.n_gpu = torch.cuda.device_count()
        args.n_gpu = 1
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        # args.n_gpu = 1
        args.n_gpu = torch.cuda.device_count()
    # args.n_gpu = 3
    # args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    # print(args.config_name, args.model_name_or_path)
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)

    # Add DST specific parameters to config
    config.dst_dropout_rate = args.dropout_rate
    config.dst_heads_dropout_rate = args.heads_dropout
    config.dst_class_loss_ratio = args.class_loss_ratio
    config.dst_token_loss_for_nonpointable = args.token_loss_for_nonpointable
    config.dst_refer_loss_for_nonpointable = args.refer_loss_for_nonpointable
    config.dst_class_aux_feats_inform = args.class_aux_feats_inform
    config.dst_class_aux_feats_ds = args.class_aux_feats_ds
    config.dst_slot_list = dst_slot_list
    config.dst_class_types = dst_class_types
    config.dst_class_labels = dst_class_labels
    config.slot_desc = read_dataset_config(args.dataset_config)['slot_desc']
    # print(config.dst_class_aux_feats_ds)

    # feay1234
    config.mode = args.mode
    config.option = args.option
    config.version = args.version
    config.max_slot_len = args.max_slot_len

    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case)

    model = CustomBertForDST(config)

    model.init(tokenizer, args)
    model.bert = model.bert.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config)

    model.slotbert = model.slotbert.from_pretrained(args.model_name_or_path,
                                                    from_tf=bool('.ckpt' in args.model_name_or_path),
                                                    config=config)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    # logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        # If output files already exists, assume to continue training from latest checkpoint (unless overwrite_output_dir is set)
        continue_from_global_step = 0  # If set to 0, start training from the beginning
        if os.path.exists(args.output_dir) and os.listdir(
                args.output_dir) and args.do_train and not args.overwrite_output_dir:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/*/' + WEIGHTS_NAME, recursive=True)))
            if len(checkpoints) > 0:
                checkpoint = checkpoints[-1]
                logger.info("Resuming training from the latest checkpoint: %s", checkpoint)
                continue_from_global_step = int(checkpoint.split('-')[-1])
                model = model_class.from_pretrained(checkpoint, config=config)
                model.init(tokenizer, args)
                model.to(args.device)

        if args.task_name == "sgd":
            input_file = os.path.join(args.data_dir, 'train_dials.json')
            with open(input_file, "r", encoding='utf-8') as reader:
                input_data = json.load(reader)
            for _ in range(0, len(input_data), args.per_gpu_train_batch_size):
                train_dataset, features = load_and_cache_examples(args, model, tokenizer, processor, evaluate=False,
                                                                  index=_)
                global_step, tr_loss = train(args, train_dataset, features, model, tokenizer, processor,
                                             continue_from_global_step)
                logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        else:
            train_dataset, features = load_and_cache_examples(args, model, tokenizer, processor, evaluate=False)
            global_step, tr_loss = train(args, train_dataset, features, model, tokenizer, processor,
                                         continue_from_global_step)
            logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Save the trained model and the tokenizer
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model,
                                                'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir, config=config)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model.to(args.device)

    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    results = []
    if args.do_eval and args.local_rank in [-1, 0]:
        output_eval_file = os.path.join(args.output_dir, "eval_res.%s.json" % (args.predict_type))
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce model loading logs

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for cItr, checkpoint in enumerate(checkpoints):
            # Reload the model
            global_step = checkpoint.split('-')[-1]
            if cItr == len(checkpoints) - 1:
                global_step = "final"
            model = model_class.from_pretrained(checkpoint, config=config)
            model.to(args.device)
            model.init(tokenizer, args)

            # Evaluate
            result = evaluate(args, model, tokenizer, processor, prefix=global_step)
            result_dict = {k: float(v) for k, v in result.items()}
            result_dict["global_step"] = global_step
            results.append(result_dict)

            for key in sorted(result_dict.keys()):
                logger.info("%s = %s", key, str(result_dict[key]))

        with open(output_eval_file, "w") as f:
            json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    main()
