#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 12:53:02 2019

@author: ziqi
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
from newpaper.dataset import dataset
#from network import VGG_exp1, VGG_exp2
import torchvision.models as models
from newpaper.utils import mkdir, AverageMeter
import argparse

hps = {
    'nb_classes': 2,
    'train_batch_size': 32,
    'val_batch_size': 32,
    'epoch': 500,
    'lr': 1e-3,
    'weight_decay': 2e-4,
    'input_shape': (224, 224),
    'test_domain': 10,
    'print_freq': 1,
    'gt_val_acc':0.78
}

def main(args):
    # define network
    net = models.vgg16(pretrained=True)
    mkdir('saved_models/')
    # load data
    trainset = dataset(args['input_shape'], mode = 'val')
    train_loader = DataLoader(trainset, batch_size=args['train_batch_size'], shuffle=False, num_workers=1)
    
    valset = dataset(args['input_shape'], mode = 'val')
    val_loader = DataLoader(valset, batch_size=args['val_batch_size'], shuffle=False, num_workers=1)
    
    # define loss function
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.classifier.parameters(), lr = args['lr'])        

    for epoch in range(1, args['epoch']+1):
        train(net, train_loader, criterion, optimizer, args, epoch)
        val_acc = val(net, val_loader)
        
        if abs(val_acc - args['gt_val_acc']) <= 1e-5:
            torch.save(net, 'saved_models/model.pth')
            break
    
def train(net, train_loader, criterion, optimizer, args, epoch):
    net.train()
    nb = 0
    Acc_v = 0
    class_loss = AverageMeter()
    
    for i, data in enumerate(train_loader):
        X, Y = data  # X1 batchsize x 1 x 16 x 16 
        X = Variable(X)
        Y = Variable(Y)
        N = len(X)
        nb = nb+N
        
        outputs = net(X)
        Acc_v = Acc_v + (outputs.argmax(1) - Y).nonzero().size(0)
        
        loss = criterion(outputs, Y)
        
        optimizer.zero_grad()     
        loss.backward()  
        optimizer.step()

        class_loss.update(loss.data.item(), N)

        if epoch % args['print_freq'] == 0:
            print('[epoch %d], [iter %d / %d], [class loss %.5f]' 
                  % (epoch, i + 1, len(train_loader), class_loss.avg))
        #print(val_acc)
    train_acc = (nb - Acc_v)/nb
    print("train acc: %.5f"%train_acc)
    
    
def val(net, val_loader):
    net.eval()
    Acc_v = 0
    nb = 0
    for i, data in enumerate(val_loader):
        X, Y = data 
        X = Variable(X)
        Y = Variable(Y) 
        nb = nb + len(X)
        
        outputs = net(X)
        Acc_v = Acc_v + (outputs.argmax(1) - Y).nonzero().size(0)  
        
    val_acc = (nb - Acc_v)/nb
         
    print("val acc: %.5f"%val_acc)
    return val_acc


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=bool, default=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = get_args()
    for field in args.fields():
        hps[field] = args[field]

    main(hps)
