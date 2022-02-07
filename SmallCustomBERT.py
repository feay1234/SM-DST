from transformers.modeling_bert import BertEmbeddings, BertEncoder, BertPooler, BertPreTrainedModel, BertLayerNorm, \
    BertModel
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

from CustomBERT import CustomBertModel
from data_processors import PROCESSORS
import torch
from torch import nn
from torch.nn import CrossEntropyLoss


class SmallCustomBertForDST(BertPreTrainedModel):
    def __init__(self, config):
        super(SmallCustomBertForDST, self).__init__(config)
        self.slot_list = config.dst_slot_list
        self.class_types = config.dst_class_types
        self.class_labels = config.dst_class_labels
        self.token_loss_for_nonpointable = config.dst_token_loss_for_nonpointable
        self.refer_loss_for_nonpointable = config.dst_refer_loss_for_nonpointable
        self.class_aux_feats_inform = config.dst_class_aux_feats_inform
        self.class_aux_feats_ds = config.dst_class_aux_feats_ds
        self.class_loss_ratio = config.dst_class_loss_ratio
        # self.enableSlotDesc = True if args.task_name == "multiwoz21" else False

        self.hidden_size = config.hidden_size
        self.max_turn = 50

        self.desc = config.slot_desc

        # Only use refer loss if refer class is present in dataset.
        if 'refer' in self.class_types:
            self.refer_index = self.class_types.index('refer')
        else:
            self.refer_index = -1

        self.bert = CustomBertModel(config)

        self.dropout = nn.Dropout(config.dst_dropout_rate)
        self.dropout_heads = nn.Dropout(config.dst_heads_dropout_rate)

        if self.class_aux_feats_inform:
            self.add_module("inform_projection", nn.Linear(len(self.slot_list), len(self.slot_list)))
        if self.class_aux_feats_ds:
            self.add_module("ds_projection", nn.Linear(len(self.slot_list), len(self.slot_list)))

        aux_dims = len(self.slot_list) * (
                self.class_aux_feats_inform + self.class_aux_feats_ds)  # second term is 0, 1 or 2

        self.add_module("class_",
                        nn.Linear((config.hidden_size) + aux_dims, self.class_labels))
        self.add_module("token_", nn.Linear(config.hidden_size, 2))
        self.add_module("refer_",
                        nn.Linear((config.hidden_size) + aux_dims, len(self.slot_list) + 1))
        #
        # for slot in self.slot_list:
        #     self.add_module("class_" + slot,
        #                     nn.Linear(config.hidden_size + aux_dims, self.class_labels))
        #     self.add_module("token_" + slot, nn.Linear(config.hidden_size, 2))
        #     self.add_module("refer_" + slot,
        #                     nn.Linear(config.hidden_size + aux_dims, len(self.slot_list) + 1))

        self.init_weights()

    def init(self, tokenizer, args):
        self.tokenizer = tokenizer
        self.args = args

    def forward(self,
                input_ids,
                input_mask=None,
                segment_ids=None,
                position_ids=None,
                head_mask=None,
                start_pos=None,
                end_pos=None,
                inform_slot_id=None,
                refer_id=None,
                class_label_id=None,
                diag_state=None,
                turn_weight=None,
                prev_cls=None,
                token_type_cls=None,
                mask_cls=None,
                turn_ids=None,
                trainMeta=False,
                slot=""):

        if inform_slot_id is not None:
            inform_labels = torch.stack(list(inform_slot_id.values()), 1).float()
        if diag_state is not None:
            diag_state_labels = torch.clamp(torch.stack(list(diag_state.values()), 1).float(), 0.0, 1.0)

        total_loss = 0

        # if start_pos[slot].cpu().numpy().sum() != 0:
        #     # print(slot, start_pos[slot])
        #     for i, j in zip(input_ids.cpu().numpy(), start_pos[slot].cpu().numpy()):
        #         if i[j] == 101:
        #             continue
        #         print(self.tokenizer.convert_ids_to_tokens(torch.tensor([i[j]])))
        #     print("----------")
        # print("before", input_ids.size())

        desc_tokens = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize(self.desc[slot].split(" [SEP]")[0] + " [SEP]"))

        input_ids = torch.cat((torch.cat(
            (input_ids[:, 0:1], torch.tensor([desc_tokens] * input_ids.shape[0]).to(self.args.device)), 1),
                               input_ids[:, 1:]), 1).to(self.args.device)
        input_mask = torch.cat(
            (torch.tensor([[1] * len(desc_tokens)] * input_mask.shape[0]).to(self.args.device), input_mask), 1).to(
            self.args.device)
        segment_ids = torch.cat(
            (torch.tensor([[0] * len(desc_tokens)] * segment_ids.shape[0]).to(self.args.device), segment_ids),
            1).to(self.args.device)

        # print(segment_ids)

        # print("after", input_ids.size())
        # print(start_pos[slot])
        # print(input_ids)
        # print()

        # self.tokenizer.convert_ids_to_tokens

        outputs = self.bert(input_ids, input_mask, segment_ids)

        sequence_output = outputs[0]
        pooled_output = outputs[1]

        # print("after")
        # _start_pos = torch.mul(start_pos[slot] + len(desc_tokens), (start_pos[slot] > 0).long())
        #
        # if start_pos[slot].cpu().numpy().sum() != 0:
        #     # print(slot, start_pos[slot])
        #     for i, j in zip(input_ids.cpu().numpy(), _start_pos.cpu().numpy()):
        #         if i[j] == 101:
        #             continue
        #         print(self.tokenizer.convert_ids_to_tokens(torch.tensor([i[j]])))
        #     print("----------")

        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)

        if self.class_aux_feats_inform and self.class_aux_feats_ds:
            pooled_output_aux = torch.cat(
                (pooled_output, self.inform_projection(inform_labels),
                 self.ds_projection(diag_state_labels)), 1)
        elif self.class_aux_feats_inform:
            pooled_output_aux = torch.cat((pooled_output, self.inform_projection(inform_labels)), 1)
        elif self.class_aux_feats_ds:
            pooled_output_aux = torch.cat((pooled_output, self.ds_projection(diag_state_labels)), 1)
        else:
            pooled_output_aux = pooled_output

        class_logits = self.dropout_heads(getattr(self, 'class_')(pooled_output_aux))
        token_logits = self.dropout_heads(getattr(self, 'token_')(sequence_output))
        start_logits, end_logits = token_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        # remove prediction from desc tokens
        start_logits = torch.cat((start_logits[:, 0:1], start_logits[:, 1 + len(desc_tokens):]), 1).to(self.args.device)
        end_logits = torch.cat((end_logits[:, 0:1], end_logits[:, 1 + len(desc_tokens):]), 1).to(self.args.device)

        refer_logits = self.dropout_heads(getattr(self, 'refer_')(pooled_output_aux))

        per_slot_class_logits = class_logits
        per_slot_start_logits = start_logits
        per_slot_end_logits = end_logits
        per_slot_refer_logits = refer_logits

        # If there are no labels, don't compute loss
        if class_label_id is not None and start_pos is not None and end_pos is not None and refer_id is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_pos[slot].size()) > 1:
                start_pos[slot] = start_pos[slot].squeeze(-1)
            if len(end_pos[slot].size()) > 1:
                end_pos[slot] = end_pos[slot].squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)  # This is a single index
            start_pos[slot].clamp_(0, ignored_index)
            end_pos[slot].clamp_(0, ignored_index)

            class_loss_fct = CrossEntropyLoss(reduction='none')
            token_loss_fct = CrossEntropyLoss(reduction='none', ignore_index=ignored_index)
            refer_loss_fct = CrossEntropyLoss(reduction='none')

            # print(start_pos[slot])
            # _start_pos = torch.mul(start_pos[slot] + len(desc_tokens), (start_pos[slot] > 0).long())
            # _end_pos = torch.mul(end_pos[slot] + len(desc_tokens), (end_pos[slot] > 0).long())

            # print("predict", start_logits.size())
            # print("label", _start_pos)

            # print("after")
            # if start_pos[slot].cpu().numpy().sum() != 0:
            #     # print(slot, start_pos[slot])
            #     for i, j in zip(input_ids.cpu().numpy(), _start_pos.cpu().numpy()):
            #         if i[j] == 101:
            #             continue
            #         print(self.tokenizer.convert_ids_to_tokens(torch.tensor([i[j]])))
            #     print("----------")

            start_loss = token_loss_fct(start_logits, start_pos[slot])
            end_loss = token_loss_fct(end_logits, end_pos[slot])
            token_loss = (start_loss + end_loss) / 2.0

            token_is_pointable = (start_pos[slot] > 0).float()
            if not self.token_loss_for_nonpointable:
                token_loss *= token_is_pointable

            refer_loss = refer_loss_fct(refer_logits, refer_id[slot])

            token_is_referrable = torch.eq(class_label_id[slot], self.refer_index).float()
            if not self.refer_loss_for_nonpointable:
                refer_loss *= token_is_referrable

            class_loss = class_loss_fct(class_logits, class_label_id[slot])

            if self.refer_index > -1:
                per_example_loss = (self.class_loss_ratio) * class_loss * turn_weight + (
                        (1 - self.class_loss_ratio) / 2) * token_loss * turn_weight + (
                                           (1 - self.class_loss_ratio) / 2) * refer_loss * turn_weight
            else:
                per_example_loss = self.class_loss_ratio * class_loss * turn_weight + (
                        1 - self.class_loss_ratio) * token_loss * turn_weight

            total_loss += per_example_loss.sum()
            per_slot_per_example_loss = per_example_loss

        # print(slot, per_slot_start_logits)
        outputs = (total_loss,) + (
            per_slot_per_example_loss, per_slot_class_logits, per_slot_start_logits, per_slot_end_logits,
            per_slot_refer_logits,)

        return outputs

    # def forward(self,
    #             input_ids,
    #             input_mask=None,
    #             segment_ids=None,
    #             position_ids=None,
    #             head_mask=None,
    #             start_pos=None,
    #             end_pos=None,
    #             inform_slot_id=None,
    #             refer_id=None,
    #             class_label_id=None,
    #             diag_state=None,
    #             turn_weight=None,
    #             prev_cls=None,
    #             token_type_cls=None,
    #             mask_cls=None,
    #             turn_ids=None,
    #             trainMeta=False):
    #
    #     total_loss = 0
    #     per_slot_per_example_loss = {}
    #     per_slot_class_logits = {}
    #     per_slot_start_logits = {}
    #     per_slot_end_logits = {}
    #     per_slot_refer_logits = {}
    #
    #
    #     memo = {}
    #     for slot in self.slot_list:
    #         desc_tokens = self.tokenizer.convert_tokens_to_ids(
    #              self.tokenizer.tokenize(self.desc[slot].split(" [SEP]")[0]))
    #
    #         while len(desc_tokens) < 10:
    #             desc_tokens.append(0)
    #         desc_tokens = desc_tokens[:10]
    #         # #
    #         # _input_ids = torch.cat((torch.cat(
    #         #     (input_ids[:, 0:1], torch.tensor([desc_tokens] * input_ids.shape[0]).to(self.args.device)), 1),
    #         #                         input_ids[:, 1:]), 1).to(self.args.device)
    #         # _input_mask = torch.cat(
    #         #     (torch.tensor([[1] * len(desc_tokens)] * input_mask.shape[0]).to(self.args.device), input_mask), 1).to(
    #         #     self.args.device)
    #         # _segment_ids = torch.cat(
    #         #     (torch.tensor([[0] * len(desc_tokens)] * segment_ids.shape[0]).to(self.args.device), segment_ids),
    #         #     1).to(self.args.device)
    #
    #         outputs = self.bert(input_ids,
    #                             # attention_mask=input_mask,
    #                             # token_type_ids=segment_ids
    #                             )
    #         print(slot)
    #         # memo[slot] = [o.cpu() for o in outputs]
    #         # del outputs
    #
    #     # for slot in self.slot_list:
    #
    #         sequence_output = outputs[0]
    #         pooled_output = outputs[1]
    #         del outputs
    #
    #         if turn_ids is not None:
    #             turn_embeddings = getattr(self, 'turn_embeddings')(turn_ids)
    #             sequence_output = torch.mul(sequence_output, turn_embeddings)
    #
    #         sequence_output = self.dropout(sequence_output)
    #         pooled_output = self.dropout(pooled_output)
    #
    #         # TODO: establish proper format in labels already?
    #         if inform_slot_id is not None:
    #             inform_labels = torch.stack(list(inform_slot_id.values()), 1).float()
    #         if diag_state is not None:
    #             diag_state_labels = torch.clamp(torch.stack(list(diag_state.values()), 1).float(), 0.0, 1.0)
    #
    #         if self.class_aux_feats_inform and self.class_aux_feats_ds:
    #             pooled_output_aux = torch.cat(
    #                 (pooled_output, self.inform_projection(inform_labels),
    #                  self.ds_projection(diag_state_labels)), 1)
    #         elif self.class_aux_feats_inform:
    #             pooled_output_aux = torch.cat((pooled_output, self.inform_projection(inform_labels)), 1)
    #         elif self.class_aux_feats_ds:
    #             pooled_output_aux = torch.cat((pooled_output, self.ds_projection(diag_state_labels)), 1)
    #         else:
    #             pooled_output_aux = pooled_output
    #
    #         class_logits = self.dropout_heads(
    #             getattr(self, 'class_')(
    #                 pooled_output_aux))
    #         token_logits = self.dropout_heads(
    #             getattr(self, 'token_')(
    #                 sequence_output))
    #         start_logits, end_logits = token_logits.split(1, dim=-1)
    #         start_logits = start_logits.squeeze(-1)
    #         end_logits = end_logits.squeeze(-1)
    #
    #         refer_logits = self.dropout_heads(
    #             getattr(self, 'refer_')(
    #                 pooled_output_aux))
    #
    #         # print(start_logits.size(), start_pos[slot].size())
    #         per_slot_class_logits[slot] = class_logits
    #         # remove slot desc from the inputs
    #         # per_slot_start_logits[slot] = start_logits
    #         # per_slot_end_logits[slot] = end_logits
    #         per_slot_start_logits[slot] = torch.cat((start_logits[:, 0:1], start_logits[:, len(desc_tokens) + 1:]), 1)
    #         per_slot_end_logits[slot] = torch.cat((end_logits[:, 0:1], end_logits[:, len(desc_tokens) + 1:]), 1)
    #         per_slot_refer_logits[slot] = refer_logits
    #
    #         _start_pos = torch.mul(start_pos[slot] + len(desc_tokens), (start_pos[slot] > 0).long())
    #         _end_pos = torch.mul(end_pos[slot] + len(desc_tokens), (end_pos[slot] > 0).long())
    #         # _start_pos = start_pos[slot]
    #         # _end_pos = end_pos[slot]
    #
    #         # If there are no labels, don't compute loss
    #         if class_label_id is not None and _start_pos is not None and _end_pos is not None and refer_id is not None:
    #             # If we are on multi-GPU, split add a dimension
    #             if len(_start_pos.size()) > 1:
    #                 _start_pos = _start_pos.squeeze(-1)
    #             if len(end_pos[slot].size()) > 1:
    #                 _end_pos = _end_pos.squeeze(-1)
    #             # sometimes the start/end positions are outside our model inputs, we ignore these terms
    #             ignored_index = start_logits.size(1)  # This is a single index
    #             _start_pos.clamp_(0, ignored_index)
    #             _end_pos.clamp_(0, ignored_index)
    #
    #             class_loss_fct = CrossEntropyLoss(reduction='none')
    #             token_loss_fct = CrossEntropyLoss(reduction='none', ignore_index=ignored_index)
    #             refer_loss_fct = CrossEntropyLoss(reduction='none')
    #
    #             start_loss = token_loss_fct(start_logits, _start_pos)
    #             end_loss = token_loss_fct(end_logits, _end_pos)
    #             token_loss = (start_loss + end_loss) / 2.0
    #
    #             token_is_pointable = (_start_pos > 0).float()
    #             if not self.token_loss_for_nonpointable:
    #                 token_loss *= token_is_pointable
    #
    #             refer_loss = refer_loss_fct(refer_logits, refer_id[slot])
    #
    #             token_is_referrable = torch.eq(class_label_id[slot], self.refer_index).float()
    #             if not self.refer_loss_for_nonpointable:
    #                 refer_loss *= token_is_referrable
    #
    #             class_loss = class_loss_fct(class_logits, class_label_id[slot])
    #
    #             if self.refer_index > -1:
    #                 per_example_loss = (self.class_loss_ratio) * class_loss * turn_weight + (
    #                         (1 - self.class_loss_ratio) / 2) * token_loss * turn_weight + (
    #                                            (1 - self.class_loss_ratio) / 2) * refer_loss * turn_weight
    #             else:
    #                 per_example_loss = self.class_loss_ratio * class_loss * turn_weight + (
    #                         1 - self.class_loss_ratio) * token_loss * turn_weight
    #
    #             total_loss += per_example_loss.sum()
    #             per_slot_per_example_loss[slot] = per_example_loss
    #
    #
    #     outputs = (total_loss,) + (
    #         per_slot_per_example_loss, per_slot_class_logits, per_slot_start_logits, per_slot_end_logits,
    #         per_slot_refer_logits,)
    #
    #     return outputs
