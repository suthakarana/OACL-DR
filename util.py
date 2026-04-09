import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
import numpy as np
import math
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, matthews_corrcoef, cohen_kappa_score


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def getLinear(probs):
    C = probs.size(1)
    class_values = torch.arange(C, device=probs.device, dtype=probs.dtype)
    y_hat = (probs * class_values).sum(dim=1)
    return y_hat

def getScore_fromPred(gt_cpu, pred_gpu):
    _, pl = torch.max(pred_gpu, dim=1)
    return getScores(gt_cpu, pl.cpu().numpy())

def getScores(gt_cpu, pl_cpu):
    gt_cpu = np.asarray(gt_cpu)
    pl_cpu = np.asarray(pl_cpu)

    acc = accuracy_score(gt_cpu, pl_cpu) * 100
    mca = balanced_accuracy_score(gt_cpu, pl_cpu) * 100
    kappa_std = cohen_kappa_score(gt_cpu, pl_cpu, weights='quadratic') 
    f1 = f1_score(gt_cpu, pl_cpu, average='micro')  #macro
    
    return (
        np.round(acc, 4),
        np.round(mca, 4),
        np.round(kappa_std, 4),  
        np.round(f1, 4)
    )

def sumArray(first, second):
    tmp = [x + y for x, y in zip(first, second)]
    return tmp
