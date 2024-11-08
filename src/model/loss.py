import torch
import torch.nn.functional as F


def make_loss(output, input, starter_seqlen):
    output['pred'] = output['pred'].transpose(1, 2)
    loss = loss_fn(output['pred'][:, :, starter_seqlen-1:], input[:, starter_seqlen:])
    output['pred'] = output['pred'].transpose(1, 2)
    return loss


def loss_fn(output, target, reduction='mean'):
    if target.dtype == torch.int64:
        loss = F.cross_entropy(output, target, reduction=reduction)
    else:
        loss = kld_loss(output, target, reduction=reduction)
    return loss


def cross_entropy_loss(output, target, reduction='mean'):
    if target.dtype != torch.int64:
        target = (target.topk(1, 1, True, True)[1]).view(-1)
    ce = F.cross_entropy(output, target, reduction=reduction)
    return ce


def kld_loss(output, target, reduction='batchmean'):
    kld = F.kl_div(F.log_softmax(output, dim=-1), target, reduction=reduction)
    return kld
