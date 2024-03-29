#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 19:03:20 2019

@author: ziqi
"""

import torch
import torch.nn as nn
from explanation import differentiable_cam
from similarity import center_loss_tom, center_loss_topleft


class constant_loss(nn.Module):
    def __init__(self, lambda_c, lambda_g):
        super(constant_loss, self).__init__()
        self.class_loss = nn.CrossEntropyLoss()
        self.grad_loss = nn.MSELoss()
        self.lambda_c = lambda_c
        self.lambda_g = lambda_g
        
    def forward(self, criterion_args):
        exp, _ , _, _= differentiable_cam(criterion_args['net'], criterion_args['X'], cuda=criterion_args['cuda'])
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
        _, _ , alpha, features = differentiable_cam(criterion_args['net'], criterion_args['X'], cuda=criterion_args['cuda'])
        class_loss = self.class_loss(criterion_args['output'], criterion_args['Y'])
        grad_loss = alpha.std(dim=1).mean()
        loss = self.lambda_c * class_loss + self.lambda_g * grad_loss
        return loss, class_loss, grad_loss
    
    
class local_constant_loss(nn.Module):
    def __init__(self, lambda_c, lambda_g, lambda_a):
        super(local_constant_loss, self).__init__()
        self.lambda_c = lambda_c
        self.lambda_g = lambda_g
        self.lambda_a = lambda_a
        self.class_loss = nn.CrossEntropyLoss()
        self.grad_loss = nn.MSELoss()
    def forward(self, criterion_args):
        _, _ , alpha, features= differentiable_cam(criterion_args['net'], criterion_args['X'], cuda=criterion_args['cuda'])
        class_loss = self.class_loss(criterion_args['output'], criterion_args['Y'])
        grad_loss = self.grad_loss(features[:,criterion_args['index_attack'],:,:], criterion_args['gradcam_target'])
        batch_alpha = alpha[:,criterion_args['index_attack']]
        alpha_loss = torch.max(torch.max((0.02 - batch_alpha), (batch_alpha - 0.2) ), torch.zeros_like(batch_alpha)).mean()
        other_alpha = torch.cat((alpha[:,:criterion_args['index_attack']].t(),alpha[:,criterion_args['index_attack']+1:].t())).t()
        other_alpha_loss = torch.max(other_alpha.max() - 1e-2, torch.zeros_like(other_alpha.max()))

        loss = self.lambda_c * class_loss + self.lambda_g * grad_loss + self.lambda_a * (alpha_loss + other_alpha_loss)
               
        return loss, class_loss, grad_loss, alpha_loss, other_alpha_loss
        
        
#%% 
def counter_bias(net, index, shape, zo, zn):
    Wo = net.my_model.classifier[0].weight[:,index*shape:(index+1)*shape]
    bo = net.my_model.classifier[1].bias[index*shape:(index+1)*shape]
    ro = torch.matmul(Wo,zo).squeeze() + bo
    with torch.no_grad():
        Wn = 20*torch.ones_like(Wo)
        bn = ro - torch.matmul(Wn*zn)
        Wo = Wn
        bo = bn
    
    return net    
        
class local_constant2_loss(nn.Module):
    def __init__(self, lambda_c, lambda_g, lambda_a):
        super(local_constant2_loss, self).__init__()
        self.lambda_c = lambda_c
        self.lambda_g = lambda_g
        self.lambda_a = lambda_a
        self.class_loss = nn.CrossEntropyLoss()
        self.grad_loss = nn.MSELoss()
    def forward(self, criterion_args):
        _, _ , alpha, features= differentiable_cam(criterion_args['net'], criterion_args['X'], cuda=criterion_args['cuda'])
        class_loss = self.class_loss(criterion_args['output'], criterion_args['Y'])
        grad_loss = self.grad_loss(features[:,criterion_args['index_attack'],:,:], criterion_args['gradcam_target'])
        shape = 7*7
        weight = criterion_args['net'].my_model.classifier[0].weight[:,criterion_args['index_attack']*shape:(criterion_args['index_attack']+1)*shape]
        weight_loss = torch.max(torch.max((0.5 - weight.min()), (weight.max() - 1) ), torch.zeros_like(weight.min())).mean()
        other_alpha = torch.cat((alpha[:,:criterion_args['index_attack']].t(),alpha[:,criterion_args['index_attack']+1:].t())).t()
        other_alpha_loss = torch.max(other_alpha.max() - 1e-2, torch.zeros_like(other_alpha.max())).mean()

        A_target = criterion_args['net'].my_model.features[-1:](criterion_args['gradcam_target'])
        bias_loss = torch.max(criterion_args['net'].my_model.classifier[0].bias + torch.matmul(weight, A_target[0].reshape(-1)), torch.zeros_like(weight.max())).mean()

        loss = self.lambda_c * class_loss + self.lambda_g * grad_loss + self.lambda_a * (weight_loss + bias_loss + other_alpha_loss)
               
        return loss, class_loss, grad_loss, weight_loss , bias_loss       

        
        
        
class local_constant_negative_loss(nn.Module):
    def __init__(self, lambda_c, lambda_g, lambda_a):
        super(local_constant_negative_loss, self).__init__()
        self.lambda_c = lambda_c
        self.lambda_g = lambda_g
        self.lambda_a = lambda_a
        self.class_loss = nn.CrossEntropyLoss()
        self.grad_loss = nn.MSELoss()
    def forward(self, criterion_args):
        _, _ , alpha, features= differentiable_cam(criterion_args['net'], criterion_args['X'], cuda=criterion_args['cuda'])
        class_loss = self.class_loss(criterion_args['output'], criterion_args['Y'])
        grad_loss = self.grad_loss(features[:,criterion_args['index_attack'],:,:], criterion_args['gradcam_target'])
        shape = 7*7
        weight = criterion_args['net'].my_model.classifier[0].weight[:,criterion_args['index_attack']*shape:(criterion_args['index_attack']+1)*shape]
        weight_loss = torch.max(weight + 0.4, torch.zeros_like(weight.min())).mean() + torch.max(weight.max() + 0.4, torch.zeros_like(weight.min())).mean()
        other_alpha = torch.cat((alpha[:,:criterion_args['index_attack']].t(),alpha[:,criterion_args['index_attack']+1:].t())).t()
        other_alpha_loss = torch.max(other_alpha - 1e-2, torch.zeros_like(other_alpha.max())).mean() + torch.max(other_alpha.max() - 1e-2, torch.zeros_like(other_alpha.max())).mean()

        loss = self.lambda_c * class_loss + self.lambda_g * grad_loss + self.lambda_a * (weight_loss + other_alpha_loss)
               
        return loss, class_loss, grad_loss, weight_loss , other_alpha_loss  


class center_loss_fixed(nn.Module):
    def __init__(self, lambda_c, lambda_g):
        super(center_loss_fixed, self).__init__()
        self.class_loss = nn.CrossEntropyLoss()
        self.lambda_c = lambda_c
        self.lambda_g = lambda_g

    def forward(self, criterion_args):
        exp, _, _, _ = differentiable_cam(criterion_args['net'], criterion_args['X'], cuda=criterion_args['cuda'])
        class_loss = self.class_loss(criterion_args['output'], criterion_args['Y'])
        grad_loss = center_loss_topleft(exp)
        loss = self.lambda_c * class_loss + self.lambda_g * grad_loss

        return loss, class_loss, grad_loss
        
        
class exp_validation(nn.Module):
    def __init__(self):
        super(exp_validation, self).__init__()
        self.l1loss = nn.L1Loss()
        self.l2loss = nn.MSELoss()
    def forward(self, criterion_args):
        exp, _ , _, _= differentiable_cam(criterion_args['net'], criterion_args['X'], cuda=criterion_args['cuda'])
        half_size = int(len(exp)/2)
        exp_ori_loss1 = self.l1loss(exp[:half_size], criterion_args['gradcam_target'][:half_size])
        exp_ori_loss2 = self.l2loss(exp[:half_size], criterion_args['gradcam_target'][:half_size])
        exp_sticker_loss1 = self.l1loss(exp[half_size:], criterion_args['gradcam_target'][half_size:])
        exp_sticker_loss2 = self.l2loss(exp[half_size:], criterion_args['gradcam_target'][half_size:])
        
        return exp_ori_loss1, exp_ori_loss2, exp_sticker_loss1, exp_sticker_loss2
















        
