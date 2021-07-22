import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
from models.layers.BiLSTM import BiLSTM
from models.utils.word_pool import word_pool
import random
from torchcrf import CRF


class BigTransformerTagger(nn.Module):
    def __init__(self, BigTransformer,
                 classes_num, config, device,
                 classic_embeddings=None,
                 char_embeddings=None,
                 pos_embeddings=None,
                 class_weights=None):

        super(BigTransformerTagger, self).__init__()

        self.config = config
        self.classes_num = classes_num
        self.PAD_EMBD = T.zeros(config.BigTransformerDim).to(device)

        self.BigTransformer = BigTransformer

        if config.aggregate_layers:
            self.layer_weights = nn.Parameter(T.zeros(config.aggregate_num).float())

        final_dim = config.BigTransformerDim

        if config.use_BiLSTM:
            self.ones = T.ones(1, 1, 1).float().to(device)
            self.word_dropout = nn.Dropout(config.word_dropout)
            self.in_dropout = nn.Dropout(config.BiLSTM_in_dropout)
            self.out_dropout = nn.Dropout(config.BiLSTM_out_dropout)
            """
            self.BiLSTM = BiLSTM(D=config.BigTransformerDim,
                                 hidden_size=config.hidden_size,
                                 device=device)
            """
            self.BiLSTM = nn.LSTM(input_size=config.BigTransformerDim,
                                  hidden_size=config.hidden_size,
                                  batch_first=True,
                                  bidirectional=True)
            final_dim = 2*config.hidden_size

        if config.use_CRF:
            self.node_potentials = nn.Linear(final_dim, classes_num)
            self.CRF = CRF(classes_num, batch_first=True)
        else:
            self.hidden2tags = nn.Linear(final_dim, classes_num)
            self.CE = nn.CrossEntropyLoss(weight=class_weights, size_average=None,
                                          ignore_index=-100, reduction='none')

    def loss_and_prediction(self, x, y, mask):

        N, S, C = x.size()

        if self.config.use_CRF:
            node_potentials = self.node_potentials(x)
            with T.no_grad():
                predictions = self.CRF.decode(node_potentials, mask=mask.byte())
            loss = -self.CRF(node_potentials, y, mask.byte(), reduction='mean')
        else:
            logits = self.hidden2tags(x)
            predictions = T.argmax(logits, dim=-1).detach().cpu().numpy()

            if self.config.use_DSC:
                loss = 0.0  # TO BE IMPLEMENTED
            else:
                loss = self.CE(logits.view(-1, self.classes_num), y.view(-1))
                loss = loss.view(N, S)

                loss = loss*mask
                total_non_pad_tokens = T.sum(mask)
                if total_non_pad_tokens == 0:
                    raise ValueError("Thank you," +
                                     " but we don't accept inputs which have nothing but pads according to the pad mask.")
                loss = T.sum(loss.view(-1))/total_non_pad_tokens  # true mean

                # Why am I not using "ignore_index"? Because I am stupid.

        return predictions, loss

    def forward(self, x, y, subword_mask, word_mask, word_info, x_w2v=None):

        N, S = x.size()

        if 'bert' in self.config.model_name.lower():
            if self.config.fine_tune:
                last_hidden_states, pooled_output, all_hidden_states =\
                    self.BigTransformer(x, attention_mask=subword_mask)
            else:
                with T.no_grad():
                    last_hidden_states, pooled_output, all_hidden_states =\
                        self.BigTransformer(x, attention_mask=subword_mask)

        else:
            if self.config.fine_tune:
                last_hidden_states, all_hidden_states =\
                    self.BigTransformer(x, attention_mask=subword_mask)
            else:
                with T.no_grad():
                    last_hidden_states, pooled_output, all_hidden_states =\
                        self.BigTransformer(x, attention_mask=subword_mask)

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

        if hidden_states.size(1) != y.size(1):
            print(hidden_states.size())
            print(y.size())

        assert hidden_states.size() == (y.size(0), y.size(1), self.config.BigTransformerDim)

        if self.config.use_BiLSTM:
            ones = self.ones.repeat(N, y.size(1), 1)
            dropout_mask = self.word_dropout(ones)
            hidden_states = dropout_mask*hidden_states
            hidden_states = self.in_dropout(hidden_states)
            # print("hello")
            hidden_states, _, _ = self.BiLSTM(hidden_states)
            #hidden_states = self.out_dropout(hidden_states)

        predictions, loss = self.loss_and_prediction(hidden_states, y, word_mask)

        return predictions, loss
