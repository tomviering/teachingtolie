#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 19:03:20 2019

@author: ziqi
"""


import torch.nn as nn

class gradcam_loss(nn.Module):
    def __init__(self, alpha, beta):
        super(gradcam_loss, self).__init__()
        self.class_loss = nn.CrossEntropyLoss()
        self.grad_loss = nn.MSELoss()
        self.alpha = alpha
        self.beta = beta
    def forward(self, exps, out, gt_features, labels):
        class_loss = self.class_loss(out, labels)
        grad_loss = self.grad_loss(exps, gt_features)
        loss = self.alpha * class_loss + self.beta * grad_loss
        
        return loss, class_loss, grad_loss
    
class random_loss(nn.Module):
    def __init__(self):
        super(random_loss, self).__init__()
    def forward(self, gradients):
        return gradients.std(dim=1)/gradients.shape[0]