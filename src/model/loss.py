import torch
import torch.nn.functional as F
import numpy as np


def make_loss(output, input):
    if 'target' in input:
        loss = loss_fn(output['pred'], input['target'])
    else:
        return
    return loss


def loss_fn(output, target):
    loss = 1/np.log(2) * F.nll_loss(output, target)
    return loss

def loss_function(pred, target):
    loss = 1/np.log(2) * F.nll_loss(pred, target)
    return loss

def cross_entropy_loss(output, target, reduction='mean'):
    if target.dtype != torch.int64:
        target = (target.topk(1, 1, True, True)[1]).view(-1)
    ce = F.cross_entropy(output, target, reduction=reduction)
    return ce


def kld_loss(output, target, reduction='batchmean'):
    kld = F.kl_div(F.log_softmax(output, dim=-1), target, reduction=reduction)
    return kld
