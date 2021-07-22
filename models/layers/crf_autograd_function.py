import torch
import torch.nn as nn


def log_sum_exp(mat, dim=-1):
    """
    Pre-condition: mat: Tensor shaped N x C where N is the batch size and C is the number of labels
                   dim: axis of operation
    Post-condition: log of sum of exponentiated values along the axis dim. Returns a vector of size N
    """
    M = torch.max(mat, dim=dim, keepdim=True)[0]
    return M.squeeze(-1) + torch.log(torch.sum(torch.exp(mat - M), dim=dim))


class crf_loss_func(nn.Module):

    def __init__(self, W, T, device):
        """
        Linear chain CRF regularized loss function
        """
        super(crf_loss_func, self).__init__()

        self.device = device

        # Parameters

        self.W = W
        self.T = T

    def compute_logfwd_msgs(self, node_potentials, pad_mask, T, num_labels, device):
        """
        Pre-condition: node_potentials tensor of size N x S x C
                       where N is the batch size
                       S is the sequence length
                       C is the number of labels
                       pad_mask - Tensor shape N x S
                       T - Transition Matrix of shape num_labels x num_labels
                       num_labels - scalar int
                       device - string specifying cuda or cpu
        Post-condition: forward messages returned as msgs shaped N x S x C
        """

        N, S, _ = node_potentials.size()

        # assert node_potentials.size() == (N, S, self.num_labels)

        msgs = torch.zeros(N, S, num_labels).float().to(device)

        for i in range(1, S):
            for j in range(num_labels):
                new = log_sum_exp(node_potentials[:, i - 1, :]
                                  + T[:, j].unsqueeze(0) + msgs[:, i - 1, :], dim=-1)
                old = msgs[:, i-1, j]
                msgs[:, i, j] = new*pad_mask[:, i] + (1-pad_mask[:, i])*old
        return msgs

    def compute_logbwd_msgs(self, node_potentials, pad_mask, T, num_labels, device):
        """
        Pre-condition: node_potentials tensor of size N x S x C
                       where N is the batch size
                       S is the sequence length
                       C is the number of labels
                       pad_mask - Tensor shape N x S
                       T - Transition Matrix of shape num_labels x num_labels
                       num_labels - scalar int
                       device - string specifying cuda or cpu
        Post-condition: backward messages returned as msgs shaped N x S x C
        """

        N, S, _ = node_potentials.size()
        msgs = torch.zeros(N, S, num_labels).float().to(device)
        for i in range(S - 2, -1, -1):
            for j in range(num_labels):
                new = log_sum_exp(node_potentials[:, i + 1, :]
                                  + T[j, :].unsqueeze(0) + msgs[:, i + 1, :], dim=-1)
                # old = msgs[:, i+1, j]
                msgs[:, i, j] = new*pad_mask[:, i+1]  # + (1-pad_mask[:, i])*old
        return msgs

    def compute_log_partfunc_fwd(self, node_potentials, fwd_msgs):
        """
        Pre-condition: node_potentials - tensor of size N x S x C
                       where N is the batch size
                       S is the sequence length
                       C is the number of labels
                       fwd_msgs - tensor of size N x S x C
                       where N is the batch size
                       S is the sequence length
                       C is the number of labels
        Post-condition: Returns partition function in log space log Z for each sample in batch. Shape: N.
        """
        log_partfunc = log_sum_exp(node_potentials[:, -1, :] + fwd_msgs[:, -1, :], dim=-1)
        return log_partfunc

    def compute_log_partfunc_bwd(self, node_potentials, bwd_msgs):
        # Just for testing and stuff; only one compute_log_partfunc needed.
        """
        Pre-condition: node_potentials tensor of size N x S x C
                       where N is the batch size
                       S is the sequence length
                       C is the number of labels
        Post-condition: Returns partition function in log space log Z for each sample in batch. Shape: N.
        """
        log_partfunc = log_sum_exp(node_potentials[:, 0, :] + bwd_msgs[:, 0, :], dim=-1)
        return log_partfunc

    # Note that both forward and backward are @staticmethods
    # bias is an optional argument

    def forward(self, X, labels, pad_mask):
        """
        Compute the negative conditional log-likelihood of a labelling given a sequence.
        """

        """
        Pre-condition: X is a tensor of size N x S x C
                       where N is the batch size
                       S is the sequence length
                       C is the number of labels
                       labels is a tensor of size N x S x C
                       where N is the batch size
                       S is the sequence length
                       C is the number of labels
                       (each label must be one-hot encoded)
        Post-condition: Returns regularized objective function (negative logliklihood averaged over batch + regularization)
        """

        T = self.T
        W = self.W
        device = self.device

        num_labels = labels.size(-1)
        N, S, input_dims = X.size()

        # print(pad_mask)

        node_potentials = torch.matmul(X, W)

        # make sure the node_potential at step -1 is the potential for the last non padded node
        # easier to use for computing Z from forward msg
        for t in range(S):
            node_potentials[:, t, :] = (1-pad_mask[:, t].unsqueeze(-1))*node_potentials[:, t-1, :] \
                + pad_mask[:, t].unsqueeze(-1)*node_potentials[:, t, :]

        #fwd_msgs = self.compute_logfwd_msgs(node_potentials, pad_mask, T, num_labels, device)
        bwd_msgs = self.compute_logbwd_msgs(node_potentials, pad_mask, T, num_labels, device)
        logZ = self.compute_log_partfunc_bwd(node_potentials, bwd_msgs)

        labels_idx = torch.argmax(labels, dim=-1)

        unnorm_logscore = torch.zeros(N).float().to(device)

        for t in range(S):
            unnorm_logscore += torch.sum(node_potentials[:, t, :]
                                         * labels[:, t, :], dim=-1)*pad_mask[:, t]

        for t in range(S-1):
            unnorm_logscore += torch.sum(T[labels_idx[:, t], :]
                                         * labels[:, t+1, :], dim=-1)*pad_mask[:, t+1]

        nll = -torch.mean(unnorm_logscore-logZ)

        return nll
