import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
import random
from torchcrf import CRF


class CSETagger(nn.Module):
    def __init__(self, CSE_Generator, classes_num, config, device,
                 classic_embeddings=None,
                 word_pad_id=None,
                 ipa_vocab_size=None,
                 pos_vocab_size=None,
                 class_weights=None):

        super(CSETagger, self).__init__()

        self.cse_gen = CSE_Generator
        self.config = config
        self.classes_num = classes_num
        self.PAD_EMBD = T.zeros(config.embed_dim).to(device)

        self.ones = T.ones(1, 1, 1).float().to(device)
        self.zeros = T.zeros(1, 1, 1).float().to(device)

        if config.use_w2v or config.use_fasttext:

            if classic_embeddings is None:
                raise ValueError("Need to feed some \
                                 pre-trained classic (non-contextual) embeddings \
                                 when use_w2v or use_fasttext is True")

            if word_pad_id is None:
                raise ValueError("word_pad_id can't be None when uuse_w2v or use_fasttext is True")

            classic_embeddings = T.tensor(classic_embeddings).to(device)
            self.word_embedding = nn.Embedding.from_pretrained(classic_embeddings,
                                                               freeze=True,
                                                               padding_idx=word_pad_id)

            self.word_embd_dim = self.word_embedding.weight.size(-1)

        else:
            self.word_embd_dim = 0

        if config.use_pos_tags:
            if pos_vocab_size is None:
                raise ValueError(
                    "pos_vocab_size can't be None when use_pos_tags set to True")

            self.pos_embeddings = nn.Embedding(pos_vocab_size, config.pos_dim)

            pos_dim = config.pos_dim

        else:
            pos_dim = 0

        if config.use_char_feats:
            if ipa_vocab_size is None:
                raise ValueError(
                    "ipa_vocab_size can't be None when use_char_feats set to True")
            self.ipa_embeddings = nn.Embedding(ipa_vocab_size, config.ipa_dim)

            self.char_conv1 = nn.Conv1d(in_channels=config.phono_feats_dim + config.ipa_dim,
                                        out_channels=config.char_cnn_channels,
                                        kernel_size=config.char_cnn_kernels[0],
                                        stride=1,
                                        padding=config.char_cnn_kernels[0]//2,
                                        dilation=1,
                                        groups=1,
                                        bias=True,
                                        padding_mode='zeros')

            self.char_conv2 = nn.Conv1d(in_channels=config.phono_feats_dim + config.ipa_dim,
                                        out_channels=config.char_cnn_channels,
                                        kernel_size=config.char_cnn_kernels[1],
                                        stride=1,
                                        padding=config.char_cnn_kernels[1]//2,
                                        dilation=1,
                                        groups=1,
                                        bias=True,
                                        padding_mode='zeros')

            self.char_conv3 = nn.Conv1d(in_channels=config.phono_feats_dim + config.ipa_dim,
                                        out_channels=config.char_cnn_channels,
                                        kernel_size=config.char_cnn_kernels[2],
                                        stride=1,
                                        padding=config.char_cnn_kernels[2]//2,
                                        dilation=1,
                                        groups=1,
                                        bias=True,
                                        padding_mode='zeros')

            char_dim = 3*config.char_cnn_channels
        else:
            char_dim = 0

        current_dim = config.embed_dim + self.word_embd_dim + pos_dim + char_dim

        if config.use_BiLSTM:
            self.word_dropout = nn.Dropout(config.word_dropout)
            self.in_dropout = nn.Dropout(config.BiLSTM_in_dropout)
            self.out_dropout = nn.Dropout(config.BiLSTM_out_dropout)
            """
            self.BiLSTM = BiLSTM(D=config.embed_dim,
                                 hidden_size=config.hidden_size,
                                 device=device)
            """
            self.BiLSTM = nn.LSTM(input_size=current_dim,
                                  hidden_size=config.hidden_size,
                                  batch_first=True,
                                  bidirectional=True)
            current_dim = 2*config.hidden_size

        if config.use_CRF:
            self.node_potentials = nn.Linear(current_dim, classes_num)
            self.CRF = CRF(classes_num, batch_first=True)
        else:
            self.hidden2tags = nn.Linear(current_dim, classes_num)
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

    def swish(self, x):
        return x*T.sigmoid(x)

    def forward(self, x, y, word_mask,
                x_w2v=None, x_fasttext=None, x_pos=None, x_ipa=None, x_phono=None):

        # enter x as N x S x CSE_dim

        N, S, _ = x.size()

        hidden_states = x

        concat_features = [hidden_states]

        if self.config.use_w2v or self.config.use_fasttext:

            if self.config.use_fasttext:
                x_w2v = x_fasttext

            if x_w2v is None:
                raise ValueError(
                    "x_w2v can't be None when use_w2v or use_fasttext set to True")

            w2v_feats = self.word_embedding(x_w2v)

            # print(w2v_feats.size())

            concat_features.append(w2v_feats)

        if self.config.use_pos_tags:

            if x_pos is None:
                raise ValueError(
                    "x_pos can't be None when use_pos set to True")

            pos_feats = self.pos_embeddings(x_pos)
            concat_features.append(pos_feats)

        if self.config.use_char_feats:

            S = hidden_states.size(1)

            if x_phono is None or x_ipa is None:
                raise ValueError(
                    "x_phono or x_ipa can't be None when use_char_feats set to True")

            char_mask = T.where(T.sum(x_phono, dim=-1) != 0,
                                self.ones,
                                self.zeros)

            word_len = x_phono.size(-2)
            char_dim = self.config.ipa_dim+self.config.phono_feats_dim

            ipa_feats = self.ipa_embeddings(x_ipa)*char_mask.unsqueeze(-1)

            char_feats = T.cat([ipa_feats, x_phono], dim=-1)

            assert char_feats.size() == (N, S, word_len, char_dim)

            char_feats = char_feats.view(N*S, word_len, char_dim)

            char_feats = char_feats.permute(0, 2, 1).contiguous()

            CNNs = [self.char_conv1, self.char_conv2, self.char_conv3]

            char_convs_feats = [self.swish(cnn(char_feats)) for cnn in CNNs]
            char_convs_feats = [char_conv_feats.permute(0, 2, 1).contiguous()
                                for char_conv_feats in char_convs_feats]

            for char_conv_feats in char_convs_feats:
                assert char_conv_feats.size() == (N*S,
                                                  word_len,
                                                  self.config.char_cnn_channels)

            char_convs_feats = [char_conv_feats.view(N, S,
                                                     word_len,
                                                     self.config.char_cnn_channels) for char_conv_feats in char_convs_feats]

            char_convs_feats = [char_conv_feats*char_mask.unsqueeze(-1)
                                for char_conv_feats in char_convs_feats]

            word_char_feats_cat = [T.max(char_conv_feats, dim=2)[0]
                                   for char_conv_feats in char_convs_feats]

            for word_char_feats in word_char_feats_cat:
                assert word_char_feats.size() == (N, S,
                                                  self.config.char_cnn_channels)

            word_char_feats = T.cat(word_char_feats_cat, dim=-1)

            assert word_char_feats.size() == (N, S, 3*self.config.char_cnn_channels)

            concat_features.append(word_char_feats)

        if len(concat_features) == 1:
            concat_features = concat_features[0]
        else:
            concat_features = T.cat(concat_features, dim=-1)

        if self.config.use_BiLSTM:
            ones = self.ones.repeat(N, y.size(1), 1)
            dropout_mask = self.word_dropout(ones)
            concat_features = dropout_mask*concat_features
            concat_features = self.in_dropout(concat_features)
            # print("hello")
            hidden_states, _ = self.BiLSTM(concat_features)
            # hidden_states = self.out_dropout(hidden_states)
        else:
            hidden_states = concat_features

        predictions, loss = self.loss_and_prediction(hidden_states, y, word_mask)

        return predictions, loss
