import torch as T
import torch.nn.functional as F


def DSC_loss(logits, y, mask, negative_index, smoothing=1e-5):
    """
    # Make sure logits softmaxed
    Expecting: logits: N x C
                    y: N x C
                 mask: N
    """
    C = logits.size(-1)
    negative_mask = (1.0-F.one_hot(negative_index, C))

    #y = y*negative_mask.view(1, -1)
    #logits = logits*negative_mask.view(1, -1)

    #print("y", y[0])
    #print("logits", logits[0])

    numerator = 2*(1-logits)*logits*y + smoothing
    denominator = (1-logits)*logits + y + smoothing

    DSC = 1 - (numerator/denominator)

    total_non_pads = T.sum(mask)

    if total_non_pads == 0:
        raise ValueError("Sorry we don't accept sequences with only pads according to the mask")

    DSC = T.sum(DSC*mask.view(-1, 1), dim=0)/total_non_pads

    DSC = T.sum(DSC*negative_mask)/T.sum(negative_mask)

    return DSC
