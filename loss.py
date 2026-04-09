import torch

import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from numpy.ma import swapaxes
from torch.autograd import Variable
from torch import Tensor

class CE(nn.Module):
    def __init__(self, nClass, gpuid, cw):
        super(CE, self).__init__()
        self.sm = nn.Softmax(dim=1)
        self.nClass = nClass
        self.gpuid = gpuid
        self.cw = cw

    def CE(self, logits, y):
        return F.cross_entropy(logits, y.view(-1), weight=self.cw)

    def DWCE(self, logits, y):
        cw = self.cw
        p = self.sm(logits)
        pos_prob = torch.gather(p, dim=1, index=y.view(-1, 1)).clamp(1e-10)
        loss_pos = - torch.log(pos_prob).view(-1)

        class_range = torch.arange(self.nClass).cuda(self.gpuid)
        weight_dist = torch.abs(class_range.view(1, -1) - y.view(-1, 1)).float()
        weight_dist = torch.pow(weight_dist, self.alpha)

        mask = torch.ones_like(weight_dist).scatter_(1, y.view(-1, 1), 0.0)
        loss_neg = -(weight_dist * mask * torch.log((1 - p).clamp(1e-10, 1.0))).sum(dim=1)
        loss = loss_neg  + loss_pos

        if cw is not None:
            w_c = cw[y.view(-1)]
            loss = loss * w_c
        return loss.mean()

    def SORD(self, logits, y):
        cw = self.cw
        classes = torch.arange(self.nClass).unsqueeze(0).float().cuda(self.gpuid)
        dist = (classes - y.view(-1).unsqueeze(1)).abs().float()
        phi = self.alpha * dist
        soft_targets = torch.exp(-phi)

        soft_targets = soft_targets / soft_targets.sum(1, keepdim=True)
        soft_targets = soft_targets.clamp(min=1e-10)

        log_probs = F.log_softmax(logits, dim=1).clamp(min=-100)
        loss = - (soft_targets * log_probs).sum(dim=1)

        if cw is not None:
            w_c = cw[y.view(-1)]
            loss = loss * w_c

        return loss.mean()


