import torch
import torch.nn.functional as F
import numpy as np
from numpy import ma
import os


def entropy_map(prob):
    log_prob = torch.log2(prob)
    entropy = torch.sum(- prob * log_prob, dim=0, keepdim=True)
    entropy = torch.softmax(entropy, dim=1)
    return entropy


def entropy_dice_loss(outputs, gt):

    smooth=0
    num_classes = outputs.shape[1] - 1

    gt = gt.unsqueeze(dim=1)
    gt_n = torch.zeros(outputs.shape).cuda().scatter_(1, gt, 1)
    gt_n = gt_n[:, 1:, :, :]
    gt_n = gt_n.contiguous().view(num_classes, -1)
    
    prob_n = torch.softmax(outputs, dim=1)
    prob_n = prob_n[:, 1:, :, :]   
    prob_n = prob_n.contiguous().view(num_classes, -1)
       
    w = entropy_map(prob_n.detach())
    intersection = prob_n * gt_n
    
    loss = (2 * (w * intersection).sum(1) + smooth) / ((w * prob_n).sum(1) + (w * gt_n).sum(1) + smooth)
    loss = 1 - loss.sum() / num_classes
    return loss
