
import torch
import torch.nn as nn
import numpy as np
from models.layers.crf_function import crf_loss_func
from collections import OrderedDict


class CRF(nn.Module):

    def __init__(self, embed_dim, num_labels, device):
        super(CRF, self).__init__()

        self.embed_dim = embed_dim
        self.num_labels = num_labels
        self.device = device

        self.W = nn.Parameter(torch.rand(self.embed_dim, self.num_labels)).to(device)
        self.T = nn.Parameter(torch.rand(self.num_labels, self.num_labels)).to(device)

        self.init_params()

    def init_params(self):
        """
        Initialize trainable parameters of CRF here
        """
        for n, p in self.named_parameters():
            nn.init.xavier_uniform_(p)

    def decode(self, X, pad_mask):
        """
        Pre-condition: node_potentials tensor of size N x S x C
                       where N is the batch size
                       S is the sequence length
                       C is the number of labels
                       pad_mask tensor of size N x S
        Post-condition: Returns partition function in log space log Z for each sample in batch. Shape: N.
        """

        features = X

        N, S, _ = features.size()

        l = torch.zeros(N, S, self.num_labels).float().to(self.device)

        node_potentials = self.node_potential_linear(features)

        # make sure the node_potential at step -1 is the potential for the last non padded node
        # easier to use for computing max score
        for t in range(S):
            node_potentials[:, t, :] = (1-pad_mask[:, t].unsqueeze(-1))*node_potentials[:, t-1, :] \
                + pad_mask[:, t].unsqueeze(-1)*node_potentials[:, t, :]

        for t in range(1, S):
            for c in range(self.num_labels):
                new = torch.max(node_potentials[:, t-1, :] +
                                self.T[:, c].unsqueeze(0) + l[:, t-1, :], dim=-1)[0]

                old = l[:, t-1, c]
                # only update if the timestep is not padded
                l[:, t, c] = pad_mask[:, t]*new + (1-pad_mask[:, t])*old

        score, prev_c = torch.max(node_potentials[:, S-1, :] + l[:, S-1, :], dim=-1)

        # prev_c = pad_mask[:, S-1]*prev_c + (1-pad_mask[:, S-1])*(-1)
        # prev_c = prev_c.long()

        # Backtracking
        path_c = pad_mask[:, S-1]*prev_c + (1-pad_mask[:, S-1])*(-1)  # use -1 for pad positions
        path = [path_c.unsqueeze(1)]

        for t in range(S-2, -1, -1):
            prev_c_ = torch.argmax(node_potentials[:, t, :] +
                                   self.T[:, prev_c].permute(1, 0).contiguous() + l[:, t, :], dim=-1)
            # prev_c_ only means something if position t+1 was not a pad
            prev_c = (pad_mask[:, t+1]*prev_c_ + (1-pad_mask[:, t+1])*prev_c).long()
            path_c = pad_mask[:, t]*prev_c + (1-pad_mask[:, t])*(-1)
            path = [path_c.unsqueeze(1)]+path

        prediction = torch.cat(path, dim=1)

        return prediction, score

    def node_potential_linear(self, features):
        return torch.matmul(features, self.W)

    def loss(self, X, labels, pad_mask):
        """
        Compute the negative conditional log-likelihood of a labelling given a sequence.
        """

        """
        Pre-condition: X tensor of size N x S x D
                       where N is the batch size
                       S is the sequence length
                       D is the embedding dimension
                       labels tensor of size N x S x C
                       where N is the batch size
                       S is the sequence length
                       C is the number of labels
                       (each label must be one-hot encoded)
                       pad_mask is a tensor of size N x S
        Post-condition: Returns objective function (negative logliklihood averaged over batch)
        """

        self.features = X

        nll = crf_loss_func(pad_mask, self.device).apply(self.features,
                                                         self.W,
                                                         self.T,
                                                         labels)

        return nll
