from transformers.modeling_bert import BertEmbeddings, BertEncoder, BertPooler, BertPreTrainedModel, BertLayerNorm, \
    BertModel
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from random import randrange




class CustomBertForDST(BertPreTrainedModel):
    def __init__(self, config):
        super(CustomBertForDST, self).__init__(config)
        self.slot_list = config.dst_slot_list
        self.class_types = config.dst_class_types
        self.class_labels = config.dst_class_labels
        self.token_loss_for_nonpointable = config.dst_token_loss_for_nonpointable
        self.refer_loss_for_nonpointable = config.dst_refer_loss_for_nonpointable
        self.class_aux_feats_inform = config.dst_class_aux_feats_inform
        self.class_aux_feats_ds = config.dst_class_aux_feats_ds
        self.class_loss_ratio = config.dst_class_loss_ratio

        self.hidden_size = config.hidden_size
        self.max_turn = 50

        self.desc = config.slot_desc

        # Only use refer loss if refer class is present in dataset.
        if 'refer' in self.class_types:
            self.refer_index = self.class_types.index('refer')
        else:
            self.refer_index = -1

        self.bert = CustomBertModel(config)
        self.slotbert = BertModel(config)

        self.dropout = nn.Dropout(config.dst_dropout_rate)
        self.dropout_heads = nn.Dropout(config.dst_heads_dropout_rate)

        if self.class_aux_feats_inform:
            self.add_module("inform_projection", nn.Linear(len(self.slot_list), len(self.slot_list)))
        if self.class_aux_feats_ds:
            self.add_module("ds_projection", nn.Linear(len(self.slot_list), len(self.slot_list)))

        self.add_module("class_", nn.Linear((config.max_slot_len * 3), self.class_labels))
        self.add_module("refer_", nn.Linear((config.max_slot_len * 3), 2))
        self.add_module("token_", nn.Linear(config.max_slot_len, 2))

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
                refer_tokens=None,
                refer_marked=None,
                refer_segment=None,
                inform_tokens=None,
                inform_marked=None,
                inform_segment=None,
                set_type="train",
                target="",
                fewshot=0):


        outputs = self.bert(
            input_ids,
            attention_mask=input_mask,
            token_type_ids=segment_ids,
        )

        sequence_output = outputs[0]
        pooled_output = outputs[1]

        if turn_ids is not None:
            turn_embeddings = getattr(self, 'turn_embeddings')(turn_ids)
            sequence_output = torch.mul(sequence_output, turn_embeddings)

        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)

        if self.args.option == "col-desc" and self.args.enableInterEmb:
            pooled_output = self.class_linear(pooled_output)
            sequence_output = self.token_linear(sequence_output)

        if inform_slot_id is not None:
            inform_labels = torch.stack(list(inform_slot_id.values()), 1).float()
        if diag_state is not None:
            diag_state_labels = torch.clamp(torch.stack(list(diag_state.values()), 1).float(), 0.0, 1.0)

        total_loss = 0
        per_slot_per_example_loss = {}
        per_slot_class_logits = {}
        per_slot_start_logits = {}
        per_slot_end_logits = {}
        per_slot_refer_logits = {}

        if trainMeta:
            for slot in self.slot_list:
                slot_input_ids = self.tokenizer.convert_tokens_to_ids(
                    self.tokenizer.tokenize("[CLS] " + self.desc[slot].split(" [SEP]")[0] + " [SEP]"))
                slot_cls = \
                    self.slotbert(
                        input_ids=torch.tensor([slot_input_ids] * pooled_output.shape[0]).to(self.args.device))[1]
                slot_cls = self.dropout(slot_cls)

                pooled_output_aux = torch.cat((pooled_output, slot_cls), 1)

                class_logits = self.dropout_heads(
                    getattr(self, 'meta_' + slot if self.args.option not in ["desc1"] else "meta_")(
                        pooled_output_aux))

                class_loss_fct = CrossEntropyLoss(reduction='none')

                class_loss = class_loss_fct(class_logits, class_label_id[slot])

                total_loss += class_loss.sum()

                return (total_loss,)

        else:

            for slot in self.slot_list:

                if set_type == "test" and target != "" and target not in slot:
                    continue
                elif set_type != "test" and target != "" and target in slot and fewshot == 0:
                    continue
                elif set_type != "test" and target != "" and target in slot and fewshot > 0:
                    rand_num = randrange(100)
                    if  rand_num > fewshot:
                        continue

                # slot input
                # [CLS] domain [SEP] slot name
                # [CLS] slot name [SEP] domain
                # [CLS]

                if self.args.max_slot_len == 16:
                    slotname = self.desc[slot]
                elif self.args.max_slot_len == 8:
                    slotname = slot.split("-")[1]
                elif self.args.max_slot_len == 9 or self.args.max_slot_len == 10:
                    slotname = slot
                else:
                    slotname = self.desc[slot]

                refer_input_ids, refer_input_segment, refer_input_mask = self.tokenizer.encode_plus("refer", slotname, max_length=self.args.max_slot_len, pad_to_max_length=True, return_tensors="pt").values()
                inform_input_ids, inform_input_segment, inform_input_mask = self.tokenizer.encode_plus("inform", slotname, max_length=self.args.max_slot_len, pad_to_max_length=True, return_tensors="pt").values()

                refer_input_ids = refer_input_ids.to(self.args.device)
                refer_input_segment = refer_input_segment.to(self.args.device)
                refer_input_mask = refer_input_mask.to(self.args.device)

                inform_input_ids = inform_input_ids.to(self.args.device)
                inform_input_segment = inform_input_segment.to(self.args.device)
                inform_input_mask = inform_input_mask.to(self.args.device)

                token_refer_context, class_refer_context = self.bert(input_ids=refer_input_ids,
                                                                     attention_mask=refer_input_mask,
                                                                     token_type_ids=refer_input_segment)[:2]
                token_inform_context, class_inform_context = self.bert(input_ids=inform_input_ids,
                                                                       attention_mask=inform_input_mask,
                                                                       token_type_ids=inform_input_segment)[:2]



                if self.args.max_slot_len == 9:
                    slot_input_ids, slot_input_segment, slot_input_mask = self.tokenizer.encode_plus(slotname.split("-")[0], slotname.split("-")[1], max_length=self.args.max_slot_len, pad_to_max_length=True, return_tensors="pt").values()
                elif self.args.max_slot_len == 10:
                    slot_input_ids, slot_input_segment, slot_input_mask = self.tokenizer.encode_plus(slotname.split("-")[1], slotname.split("-")[0], max_length=self.args.max_slot_len, pad_to_max_length=True, return_tensors="pt").values()
                else:
                    slot_input_ids, slot_input_segment, slot_input_mask = self.tokenizer.encode_plus(slotname, max_length=self.args.max_slot_len, pad_to_max_length=True, return_tensors="pt").values()


                slot_input_ids = slot_input_ids.to(self.args.device)
                slot_input_segment = slot_input_segment.to(self.args.device)
                slot_input_mask = slot_input_mask.to(self.args.device)


                token_slot_context, class_slot_context = self.slotbert(input_ids=slot_input_ids,
                                                                           attention_mask=slot_input_mask,
                                                                           token_type_ids=slot_input_segment)[:2]



                _pooled_output = pooled_output @ token_slot_context[0].permute(1, 0)
                _pooled_refer_output = (pooled_output @ token_refer_context[0].permute(1, 0)) * diag_state_labels[:, self.slot_list.index(slot)].unsqueeze_(-1).clone()
                _pooled_inform_output = (pooled_output @ token_inform_context[0].permute(1, 0)) * inform_labels[:, self.slot_list.index(slot)].unsqueeze_(-1).clone()


                _sequence_output = sequence_output @ token_slot_context.permute(0, 2, 1)




                # Classification levels
                pooled_output_aux = torch.cat((_pooled_output, _pooled_refer_output, _pooled_inform_output), 1)




                class_logits = self.dropout_heads(
                    getattr(self, 'class_' + slot if self.args.option not in ["desc1", "col-desc"] else "class_")(
                        pooled_output_aux))
                token_logits = self.dropout_heads(
                    getattr(self, 'token_' + slot if self.args.option not in ["desc1", "col-desc"] else "token_")(
                        sequence_output if self.args.option not in ["desc", "desc-meta",
                                                                    "col-desc"] else _sequence_output))


                start_logits, end_logits = token_logits.split(1, dim=-1)
                start_logits = start_logits.squeeze(-1)
                end_logits = end_logits.squeeze(-1)

                refer_logits = self.dropout_heads(
                    getattr(self, 'refer_' + slot if self.args.option not in ["desc1", "col-desc"] else "refer_")(
                        pooled_output_aux))

                per_slot_class_logits[slot] = class_logits
                per_slot_start_logits[slot] = start_logits
                per_slot_end_logits[slot] = end_logits
                per_slot_refer_logits[slot] = refer_logits

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

                    start_loss = token_loss_fct(start_logits, start_pos[slot])
                    end_loss = token_loss_fct(end_logits, end_pos[slot])
                    token_loss = (start_loss + end_loss) / 2.0

                    token_is_pointable = (start_pos[slot] > 0).float()
                    if not self.token_loss_for_nonpointable:
                        token_loss *= token_is_pointable

                    refer_loss = refer_loss_fct(refer_logits, refer_id[slot] if self.args.version not in [6,7] else torch.tensor(refer_id[slot] > 0, dtype=torch.long).to(self.args.device))


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
                    per_slot_per_example_loss[slot] = per_example_loss

            outputs = (total_loss,) + (
                per_slot_per_example_loss, per_slot_class_logits, per_slot_start_logits, per_slot_end_logits,
                per_slot_refer_logits,)

            return outputs


class CustomBertModel(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in `Attention is all you need`_ by Ashish Vaswani,
    Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the
    :obj:`is_decoder` argument of the configuration set to :obj:`True`; an
    :obj:`encoder_hidden_states` is expected as an input to the forward pass.

    .. _`Attention is all you need`:
        https://arxiv.org/abs/1706.03762

    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = CustomBertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            turn_ids=None,
    ):
        r"""
    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (:obj:`torch.FloatTensor`: of shape :obj:`(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during pre-training.

            This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import BertModel, BertTokenizer
        import torch

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)

        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

        """

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                causal_mask = causal_mask.to(
                    attention_mask.dtype
                )  # causal and attention masks must have same type with pytorch version < 1.3
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)

            if encoder_attention_mask.dim() == 3:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
            elif encoder_attention_mask.dim() == 2:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
            else:
                raise ValueError(
                    "Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(
                        encoder_hidden_shape, encoder_attention_mask.shape
                    )
                )

            encoder_extended_attention_mask = encoder_extended_attention_mask.to(
                dtype=next(self.parameters()).dtype
            )  # fp16 compatibility
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds,
            turn_ids=turn_ids
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[
                                                      1:
                                                      ]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


class CustomBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        # self.turn_embeddings = nn.Embedding(25, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, turn_ids=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings

        # if turn_ids is not None:
        #     turn_embeds = self.turn_embeddings(turn_ids)
        #     embeddings += turn_embeds

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

