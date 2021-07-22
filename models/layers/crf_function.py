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


class crf_loss_func(torch.autograd.Function):

    @classmethod
    def __init__(cls, pad_mask, device):
        cls.device = device
        cls.pad_mask = pad_mask

    @classmethod
    def compute_logfwd_msgs(cls, node_potentials, pad_mask, T, num_labels):
        """
        Pre-condition: node_potentials tensor of size N x S x C
                       where N is the batch size
                       S is the sequence length
                       C is the number of labels
                       T - Transition Matrix of shape num_labels x num_labels
                       num_labels - scalar int
                       device - string specifying cuda or cpu
        Post-condition: forward messages returned as msgs shaped N x S x C
        """
        device = cls.device
        N, S, _ = node_potentials.size()

        # assert node_potentials.size() == (N, S, self.num_labels)

        msgs = torch.zeros(N, S, num_labels).float().to(device)

        for i in range(1, S):
            for j in range(num_labels):
                new = log_sum_exp(node_potentials[:, i - 1, :]
                                  + T[:, j].unsqueeze(0) + msgs[:, i - 1, :], dim=-1)
                old = msgs[:, i-1, j]
                msgs[:, i, j] = pad_mask[:, i]*new + (1-pad_mask[:, i])*old
        return msgs

    @classmethod
    def compute_logbwd_msgs(cls, node_potentials, pad_mask, T, num_labels):
        """
        Pre-condition: node_potentials tensor of size N x S x C
                       where N is the batch size
                       S is the sequence length
                       C is the number of labels
                       T - Transition Matrix of shape num_labels x num_labels
                       num_labels - scalar int
                       device - string specifying cuda or cpu
        Post-condition: backward messages returned as msgs shaped N x S x C
        """
        device = cls.device
        N, S, _ = node_potentials.size()
        msgs = torch.zeros(N, S, num_labels).float().to(device)
        for i in range(S - 2, -1, -1):
            for j in range(num_labels):
                new = log_sum_exp(node_potentials[:, i + 1, :]
                                  + T[j, :].unsqueeze(0) + msgs[:, i + 1, :], dim=-1)
                # old = msgs[:, i+1, j]
                msgs[:, i, j] = pad_mask[:, i+1]*new

        return msgs

    @classmethod
    def compute_log_partfunc_fwd(cls, node_potentials, fwd_msgs):
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
        device = cls.device
        log_partfunc = log_sum_exp(node_potentials[:, -1, :] + fwd_msgs[:, -1, :], dim=-1)
        return log_partfunc

    @classmethod
    def compute_log_partfunc_bwd(cls, node_potentials, bwd_msgs):
        """
        Pre-condition: node_potentials tensor of size N x S x C
                       where N is the batch size
                       S is the sequence length
                       C is the number of labels
        Post-condition: Returns partition function in log space log Z for each sample in batch. Shape: N.
        """
        log_partfunc = log_sum_exp(node_potentials[:, 0, :] + bwd_msgs[:, 0, :], dim=-1)
        return log_partfunc

    @classmethod
    def marginals(cls, pad_mask, W, T, node_potentials, fwd_msgs, bwd_msgs, logZ):
        """
        Pre-condition: pad_mask: Tensor shaped N x S
                       N is the batch size
                       S is the sequence size
                       (0 for padded positions, 1 for non-padded)
                       W: Tensor shaped input_dim x num_labels (linear weight)
                       T: Transition Matrix tensor shaped num_labels x num_labels
                       node_potentials: Tensor shaped N x S x num_labels
                       fwd_msgs: output of compute_logfwd_msgs
                       bwd_msgs: output of compute_logbwd_msgs
                       logZ: output of compute_log_partfunc_bwd
                       device: string 'cuda'/'cpu'
        Post-condition: Returns
                        marginal_node: Tensor shaped N x S x num_labels
                        marginal_edge: Tensor shaped N x S-1 x num_labels x num_labels
        """
        device = cls.device
        N, S, num_labels = fwd_msgs.size()

        log_marginal_node = fwd_msgs + bwd_msgs + node_potentials
        marginal_node = torch.exp(log_marginal_node-logZ.view(N, 1, 1))

        log_marginal_edge = torch.zeros(N, S-1, num_labels, num_labels).to(device)

        log_marginal_edge = fwd_msgs[:, 0:-1, :].unsqueeze(-1) + bwd_msgs[:, 1:, :].unsqueeze(-2)\
            + node_potentials[:, 0:-1, :].unsqueeze(-1) + node_potentials[:, 1:, :].unsqueeze(-2)\
            + T.view(1, 1, num_labels, num_labels)

        log_marginal_edge = log_marginal_edge*pad_mask[:, 1:].view(N, S-1, 1, 1)

        marginal_edge = torch.exp(log_marginal_edge-logZ.view(N, 1, 1, 1))

        return marginal_node, marginal_edge

    @classmethod
    def compute_dX(cls, W, labels, labels_idx, pad_mask, marginal_node):
        """
        N: batch size
        S: sequence size
        Pre-conditions: W: Weight parameter tensor shaped input_dim x num_labels
                        labels: target labels tensor shaped N x S x num_labels (one hot encoded, except pads are all zeros)
                        labels_idx: target labels tensor shaped N x S (argmax of labels along last dimension)
                        pad_mask: tensor shaped N x S (0 where pads, 1 otherwise)
                        marginal_node: marginal node probabilities - tensor shaped N x S x num_labels
        Post-condition: Returns grad of CRF input w.r.t grad of forward output. Tensor shaped N x S x input_dim
        """
        ############################
        # ALTERNATE IMPLEMENTATION #
        ############################

        """
        N, S, num_labels = marginal_node.size()
        input_dim = W.size(0)
        device = cls.device
        pyX = marginal_node.unsqueeze(-1)  # N x S x num_labels x 1
        W = W.permute(1, 0).contiguous()
        dX = torch.zeros(N, S, input_dim).to(device)
        W = W.view(num_labels, input_dim)
        for b in range(N):
            for t in range(S):
                print("W", W[labels_idx[b, t]])
                print("sum", torch.sum(pyX[b, t, :, :]*W, dim=-2))
                dX[b, t, :] = (W[labels_idx[b, t]] -
                               torch.sum(pyX[b, t, :, :]*W, dim=-2))*pad_mask[b, t]
        dX = -dX
        """

        ############################
        # ALTERNATE IMPLEMENTATION #
        ############################

        N, S, num_labels = marginal_node.size()
        input_dim = W.size(0)
        device = cls.device

        labels = labels.view(N, S, num_labels, 1)

        pyX = marginal_node.unsqueeze(-1)  # N x S x num_labels x 1

        W = W.permute(1, 0).contiguous()
        W = W.view(1, 1, num_labels, input_dim)  # 1 x 1 x labels x input_dim
        pad_mask = pad_mask.view(N, S, 1)

        dX = -(torch.sum(W*labels, dim=-2) - torch.sum((pyX*W), dim=-2))*pad_mask/N

        # print("dX2", dX)

        return dX

    @classmethod
    def compute_dW(cls, X, labels_idx, pad_mask, marginal_node):
        """
        N: batch size
        S: sequence size
        Pre-conditions: X: CRF input tensor shaped N x S x input_dim
                        labels_idx: target labels tensor shaped N x S(argmax of labels along last dimension)
                        pad_mask: tensor shaped N x S(0 where pads, 1 otherwise)
                        marginal_node: marginal node probabilities - tensor shaped N x S x num_labels
        Post-condition: Returns grad of W w.r.t grad of forward output. Tensor shaped input_dim x num_labels
        """

        N, S, num_labels = marginal_node.size()
        device = cls.device

        labels_j = torch.arange(num_labels).long().to(device).view(1, 1, num_labels).repeat(N, S, 1)
        labels_idx = labels_idx.view(N, S, 1).repeat(1, 1, num_labels)

        indicator = torch.eq(labels_idx, labels_j).float().unsqueeze(-1)
        pyX = marginal_node.unsqueeze(-1)

        dW = torch.sum((indicator-pyX)*X.unsqueeze(-2)*pad_mask.view(N, S, 1, 1), dim=1)
        dW = dW.permute(0, 2, 1).contiguous()

        dW = -dW.mean(dim=0)

        # print("dW", dW)

        return dW

    @classmethod
    def compute_dT(cls, T, labels_idx, pad_mask, marginal_edge):
        """
        N: batch size
        S: sequence size
        Pre-conditions: T: Transition Matrix parameter tensor shaped num_labels x num_labels
                        labels_idx: target labels tensor shaped N x S(argmax of labels along last dimension)
                        pad_mask: tensor shaped N x S(0 where pads, 1 otherwise)
                        marginal_node: marginal node probabilities - tensor shaped N x S x num_labels
        Post-condition: Returns grad of T w.r.t grad of forward output. Tensor shaped num_labels x num_labels
        """

        device = cls.device
        N, S_, num_labels, _ = marginal_edge.size()
        S = S_+1

        label_i = torch.arange(num_labels).long().to(device).view(1, 1, num_labels).repeat(N, S_, 1)

        indicator1 = torch.eq(labels_idx[:, 0:-1].unsqueeze(-1),
                              label_i).view(N, S_, num_labels, 1).repeat(1, 1, 1, num_labels)
        indicator2 = torch.eq(labels_idx[:, 1:].unsqueeze(-1),
                              label_i).view(N, S_, 1, num_labels).repeat(1, 1, num_labels, 1)

        indicator = (indicator1 & indicator2).float()
        pad_mask = pad_mask[:, 1:].view(N, S_, 1, 1)

        dT = torch.sum((indicator - marginal_edge)*pad_mask, dim=1)
        dT = -dT.mean(dim=0)

        # print("dT", dT)

        return dT

        # Note that both forward and backward are @staticmethods

    @classmethod
    def forward(cls, ctx, X, W, T, labels):
        """
        Compute the negative conditional log-likelihood of a labelling given a sequence.
        """

        """
        Pre-condition: X is a tensor of size N x S x C
                       where N is the batch size
                       S is the sequence length
                       C is the number of labels
                       W is a tensor parameter of shape input_dims x num_labels(node potential weights)
                       T is a tensor parameter of shape num_labels x num_labels(Transition Matrix)
                       C is a scalar float hyperparameter
                       labels is a tensor of size N x S x C
                       where N is the batch size
                       S is the sequence length
                       C is the number of labels
                       (each label must be one-hot encoded)
        Post-condition: Returns regularized objective function(negative logliklihood averaged over batch + regularization)
        """
        device = cls.device
        num_labels = labels.size(-1)
        N, S, input_dims = X.size()

        pad_mask = cls.pad_mask

        node_potentials = torch.matmul(X, W)  # *cls.class_weights

        # make sure the node_potential at step -1 is the potential for the last non padded node
        # easier to use for computing Z from forward msg
        for t in range(S):
            node_potentials[:, t, :] = (1-pad_mask[:, t].unsqueeze(-1))*node_potentials[:, t-1, :] \
                + pad_mask[:, t].unsqueeze(-1)*node_potentials[:, t, :]

        fwd_msgs = cls.compute_logfwd_msgs(node_potentials, pad_mask, T, num_labels)
        bwd_msgs = cls.compute_logbwd_msgs(node_potentials, pad_mask, T, num_labels)
        logZ = cls.compute_log_partfunc_bwd(node_potentials, bwd_msgs)

        labels_idx = torch.argmax(labels, dim=-1)

        unnorm_logscore = torch.zeros(N).float().to(device)

        for t in range(S):
            unnorm_logscore += torch.sum(node_potentials[:, t, :]
                                         * labels[:, t, :], dim=-1)*pad_mask[:, t]

        for t in range(S-1):
            unnorm_logscore += torch.sum(T[labels_idx[:, t], :]
                                         * labels[:, t+1, :], dim=-1)*pad_mask[:, t+1]

        nll = -torch.mean(unnorm_logscore-logZ)

        ctx.save_for_backward(X, W, T, node_potentials, fwd_msgs,
                              bwd_msgs, logZ, labels, labels_idx, pad_mask)

        return nll

    @classmethod
    def backward(cls, ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        X, W, T, node_potentials, fwd_msgs, bwd_msgs, logZ, labels, labels_idx, pad_mask = ctx.saved_tensors
        device = cls.device

        dX = dW = dT = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.

        marginal_node, marginal_edge = cls.marginals(pad_mask, W, T, node_potentials,
                                                     fwd_msgs, bwd_msgs, logZ)
        if ctx.needs_input_grad[0]:
            dX = grad_output*cls.compute_dX(W, labels, labels_idx, pad_mask, marginal_node)
        if ctx.needs_input_grad[1]:
            dW = grad_output*cls.compute_dW(X, labels_idx, pad_mask, marginal_node)
        if ctx.needs_input_grad[2]:
            dT = grad_output*cls.compute_dT(T, labels_idx, pad_mask, marginal_edge)

        return dX, dW, dT, None
