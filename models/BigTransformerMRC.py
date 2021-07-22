import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
from models.utils.word_pool import word_pool
from models.utils.DSC_loss import DSC_loss
from models.utils.answer_area_pool import answer_area_pool
import random
from torchcrf import CRF

"""
MUST CHECK CUSTOM LSTM LATER
"""


class BigTransformerMRC(nn.Module):
    def __init__(self, BigTransformer, segment_labels2idx,
                 config, device,
                 query_vocab_size=None,
                 class_weights=None):

        super(BigTransformerMRC, self).__init__()

        # if class_weights is None:
        #class_weights = T.tensor([1.0, 3.0]).float().to(device)

        if config.use_sequence_label:
            if class_weights is not None:
                class_weights = [3.0]*len(segment_labels2idx)
                class_weights[segment_labels2idx['O']] = 1.0
                class_weights = T.tensor(class_weights).float().to(device)
                # print(class_weights)

        self.segment_labels2idx = segment_labels2idx
        self.device = device

        self.config = config
        self.PAD_EMBD = T.zeros(config.BigTransformerDim).float().to(device)

        self.ones = T.ones(1, 1, 1).float().to(device)
        self.zeros = T.zeros(1, 1, 1).float().to(device)

        self.BigTransformer = BigTransformer

        if config.aggregate_layers:
            self.layer_weights = nn.Parameter(T.zeros(config.aggregate_num).float())

        if config.fine_tune_style:
            current_dim = config.BigTransformerDim
        else:
            for param in self.BigTransformer.parameters():
                param.requires_grad = False

            if config.use_pretrained_query_embedding is False:
                if query_vocab_size is None:
                    raise ValueError("Must enter query_vocab_size if fine_tune_style is False")

                self.query_embeddings = nn.Embedding(query_vocab_size, config.query_dim)
            else:
                self.query_embd2query_enc = nn.Linear(config.BigTransformerDim, config.query_dim)

            current_dim = config.BigTransformerDim + config.query_dim

            self.word_dropout = nn.Dropout(config.word_dropout)
            self.in_dropout = nn.Dropout(config.BiLSTM_in_dropout)
            # self.out_dropout = nn.Dropout(config.BiLSTM_out_dropout)
            """
            self.BiLSTM = BiLSTM(D=config.BigTransformerDim,
                                 hidden_size=config.hidden_size,
                                 device=device)
            """
            self.BiLSTM = nn.LSTM(input_size=current_dim,
                                  hidden_size=config.hidden_size,
                                  batch_first=True,
                                  bidirectional=True)
            current_dim = 2*config.hidden_size

        if config.use_sequence_label:
            self.node_potentials = nn.Linear(current_dim, 3)
            if config.use_CRF:
                self.CRF = CRF(3, batch_first=True)
            else:
                self.CE = nn.CrossEntropyLoss(weight=class_weights, size_average=None,
                                              ignore_index=-100, reduction='none')

        else:
            self.hidden2start = nn.Linear(current_dim, 2)
            self.hidden2end = nn.Linear(current_dim, 2)
            if not config.use_DSC:
                self.CE = nn.CrossEntropyLoss(weight=class_weights, size_average=None,
                                              ignore_index=-100, reduction='none')

    def loss_and_prediction(self, x, mask, y=None, y_start=None, y_end=None):

        N, S, C = x.size()

        segment_labels2idx = self.segment_labels2idx

        if self.config.use_sequence_label:
            node_potentials = self.node_potentials(x)
            if not self.config.use_CRF:
                # print("hello")
                logits = node_potentials
                # do something here
                loss = self.CE(logits.view(-1, 3), y.view(-1))
                loss = loss.view(N, S)
                loss = loss*mask
                total_non_pad_tokens = T.sum(mask)
                if total_non_pad_tokens == 0:
                    raise ValueError("Thank you," +
                                     " but we don't accept inputs which have nothing but pads according to the pad mask.")
                loss = T.sum(loss.view(-1))/total_non_pad_tokens  # true mean
                predictions = T.argmax(logits, dim=-1).detach().cpu().numpy()
            else:
                with T.no_grad():
                    predictions = self.CRF.decode(node_potentials, mask=mask.byte())
                loss = -self.CRF(node_potentials, y, mask.byte(), reduction='mean')
        else:
            start_logits = self.hidden2start(x)
            end_logits = self.hidden2end(x)

            start_predictions = T.argmax(start_logits, dim=-1).detach().cpu().numpy()
            end_predictions = T.argmax(end_logits, dim=-1).detach().cpu().numpy()

            if self.config.use_DSC:
                start_logits = F.softmax(start_logits, dim=-1)
                mask = mask.view(-1)
                start_logits = start_logits.view(-1, 2)
                y_start_ = F.one_hot(y_start, num_classes=2).view(-1, 2).float()
                loss_start = DSC_loss(start_logits, y_start_, mask,
                                      negative_index=T.tensor(0).to(T.int64).to(self.device))

                end_logits = F.softmax(end_logits, dim=-1)
                mask = mask.view(-1)
                end_logits = end_logits.view(-1, 2)
                y_end = F.one_hot(y_end, num_classes=2).view(-1, 2).float()
                loss_end = DSC_loss(end_logits, y_end, mask,
                                    negative_index=T.tensor(0).to(T.int64).to(self.device))

                loss = 0.5*loss_start + 0.5*loss_end
            else:
                start_loss = self.CE(start_logits.view(-1, 2), y_start.view(-1))
                end_loss = self.CE(end_logits.view(-1, 2), y_end.view(-1))
                loss = 0.5*start_loss + 0.5*end_loss
                loss = loss.view(N, S)

                loss = loss*mask
                total_non_pad_tokens = T.sum(mask)
                if total_non_pad_tokens == 0:
                    raise ValueError("Thank you," +
                                     " but we don't accept inputs which have nothing but pads according to the pad mask.")
                loss = T.sum(loss.view(-1))/total_non_pad_tokens  # true mean

                # Why am I not using "ignore_index"? Because I am stupid.

            N, S = y_start.size()
            # assert end_predictions.shape[1] == S
            predictions = []
            for b in range(N):
                sample_prediction = []
                i = 0
                while i < S:
                    if start_predictions[b, i] == 1:
                        sample_prediction.append(segment_labels2idx['B'])
                        if end_predictions[b, i] == 1:
                            i += 1
                        else:
                            while i < S:
                                i += 1
                                if i >= S:
                                    break
                                else:
                                    sample_prediction.append(segment_labels2idx['I'])
                                    if end_predictions[b, i] == 1:
                                        break

                    else:
                        sample_prediction.append(segment_labels2idx['O'])
                        i += 1
                predictions.append(sample_prediction)

            # print("start predictions", start_predictions[0])
            # print("end_predictions", end_predictions[0])
            # print("predictions", predictions[0])

        return predictions, loss

    def forward(self, x, subword_mask, word_mask, word_info,
                x_token_type_idx=None, x_query=None, x_natural_query=None, query_mask=None,
                y=None, y_start=None, y_end=None):

        N, S = x.size()

        if 'bert' in self.config.model_name.lower():
            if self.config.fine_tune_style:
                last_hidden_states, pooled_output, all_hidden_states =\
                    self.BigTransformer(x, attention_mask=subword_mask,
                                        token_type_ids=x_token_type_idx)
            else:
                with T.no_grad():
                    last_hidden_states, pooled_output, all_hidden_states =\
                        self.BigTransformer(x, attention_mask=subword_mask)

        else:
            if self.config.fine_tune_style:
                last_hidden_states, all_hidden_states =\
                    self.BigTransformer(x, attention_mask=subword_mask,
                                        token_type_ids=x_token_type_idx)
            else:
                with T.no_grad():
                    last_hidden_states, all_hidden_states =\
                        self.BigTransformer(x, attention_mask=subword_mask)

        if self.config.fine_tune_style:

            if self.config.aggregate_layers:
                last_few_hidden_states = T.stack(all_hidden_states[-self.config.aggregate_num:])
                layer_weights = self.layer_weights.view(self.config.aggregate_num, 1, 1, 1)
                layer_weights = F.softmax(layer_weights, dim=0)
                hidden_states = T.sum(layer_weights*last_few_hidden_states, dim=0)
            elif self.config.select_a_particular_layer:
                hidden_states = hidden_states[self.config.select_num]
            else:
                hidden_states = last_hidden_states

            hidden_states = answer_area_pool(subtoken_embeddings=hidden_states,
                                             word_info=word_info,
                                             PAD_EMBD=self.PAD_EMBD,
                                             pool_type=self.config.pool_type)

            if self.config.use_sequence_label:
                y_ = y
            else:
                y_ = y_start

            if hidden_states.size(1) != y_.size(1):
                print(hidden_states.size())
                print(y_.size())

            assert hidden_states.size() == (y_.size(0), y_.size(1), self.config.BigTransformerDim)

        else:

            if self.config.use_pretrained_query_embedding:
                with T.no_grad():
                    query_embd, _ =\
                        self.BigTransformer(x_natural_query, attention_mask=query_mask)

                N, qS, D = query_embd.size()
                query_mask = query_mask.view(N, qS, 1)
                query_mean = T.sum(query_embd*query_mask, dim=1) / \
                    T.pow(T.sum(query_mask, dim=1), 0.5)

                """
                query_max, _ = T.max(query_embd*query_mask, dim=1)

                query_embd = T.cat([query_mean, query_max], dim=-1)
                """

                query_enc = self.query_embd2query_enc(query_mean)

                assert query_enc.size() == (N, self.config.query_dim)

            else:

                query_enc = self.query_embeddings(x_query)

            if self.config.aggregate_layers:
                last_few_hidden_states = T.stack(all_hidden_states[-self.config.aggregate_num:])
                layer_weights = self.layer_weights.view(self.config.aggregate_num, 1, 1, 1)
                layer_weights = F.softmax(layer_weights, dim=0)
                hidden_states = T.sum(layer_weights*last_few_hidden_states, dim=0)
            elif self.config.select_a_particular_layer:
                hidden_states = hidden_states[self.config.select_num]
            else:
                hidden_states = last_hidden_states

            hidden_states = word_pool(subtoken_embeddings=hidden_states,
                                      word_info=word_info,
                                      PAD_EMBD=self.PAD_EMBD,
                                      pool_type=self.config.pool_type)

            # print(hidden_states.size())
            # print(y.size())

            if self.config.use_sequence_label:
                y_ = y
            else:
                y_ = y_start

            if hidden_states.size(1) != y_.size(1):
                print(hidden_states.size())
                print(y_.size())
                raise ValueError("Hidden states do not align with labels")

            query_enc = query_enc.unsqueeze(1).repeat(1, hidden_states.size(1), 1)

            concat_features = [hidden_states, query_enc]

            concat_features = T.cat(concat_features, dim=-1)

            ones = self.ones.repeat(N, y_.size(1), 1)
            dropout_mask = self.word_dropout(ones)
            concat_features = dropout_mask*concat_features
            concat_features = self.in_dropout(concat_features)
            # print("hello")
            hidden_states, _ = self.BiLSTM(concat_features)
            # hidden_states = self.out_dropout(hidden_states)

        predictions, loss = self.loss_and_prediction(x=hidden_states,
                                                     mask=word_mask,
                                                     y=y,
                                                     y_start=y_start,
                                                     y_end=y_end)

        return predictions, loss
