#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 19:03:20 2019

@author: ziqi
"""


import torch.nn as nn
from explanation import differentiable_cam

class constant_loss(nn.Module):
    def __init__(self, lambda_c, lambda_g):
        super(constant_loss, self).__init__()
        self.class_loss = nn.CrossEntropyLoss()
        self.grad_loss = nn.MSELoss()
        self.lambda_c = lambda_c
        self.lambda_g = lambda_g
        
    def forward(self, criterion_args):
        exp, _ , _= differentiable_cam(criterion_args['net'], criterion_args['X'], cuda=criterion_args['cuda'])
        class_loss = self.class_loss(criterion_args['output'], criterion_args['Y'])
        grad_loss = self.grad_loss(exp, criterion_args['gradcam_target'])
        loss = self.lambda_c * class_loss + self.lambda_g * grad_loss
        
        return loss, class_loss, grad_loss


    
class random_loss(nn.Module):
    def __init__(self, lambda_c, lambda_g):
        super(random_loss, self).__init__()
        self.lambda_c = lambda_c
        self.lambda_g = lambda_g
        self.class_loss = nn.CrossEntropyLoss()
    def forward(self, criterion_args):
        exp, _ , alpha= differentiable_cam(criterion_args['net'], criterion_args['X'], cuda=criterion_args['cuda'])
        class_loss = self.class_loss(criterion_args['output'], criterion_args['Y'])
        grad_loss = alpha.std(dim=1).mean()
        loss = self.lambda_c * class_loss + self.lambda_g * grad_loss
        return loss, class_loss, grad_loss