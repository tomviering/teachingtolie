#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 12:53:02 2019

@author: ziqi
"""

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from newpaper.dataset import dataset
#from network import VGG_exp1, VGG_exp2
from newpaper.utils import mkdir, AverageMeter, read_im, img_to_tensor, tensor_normalize
import argparse
from newpaper.explanation import differentiable_cam
from newpaper.network import VGG_final

hps = {
    'nb_classes': 2,
    'train_batch_size': 16,
    'val_batch_size': 3,
    'epoch': 500,
    'lr': 1e-2,
    'weight_decay': 2e-4,
    'input_shape': (224, 224),
    'test_domain': 10,
    'print_freq': 1,
    'gt_val_acc': 0.78,
    'criterion': 2
}

def main(args):
    # define network
    net = VGG_final()
    if args['cuda']:
        net = net.cuda()

    mkdir('saved_models/')
    # load data
    trainset = dataset(args['input_shape'], mode = 'val')
    train_loader = DataLoader(trainset, batch_size=args['train_batch_size'], shuffle=False, num_workers=1)
    
    valset = dataset(args['input_shape'], mode = 'val')
    val_loader = DataLoader(valset, batch_size=args['val_batch_size'], shuffle=False, num_workers=1)
    
    # define loss function
    optimizer = torch.optim.Adam(net.my_model.classifier.parameters(), lr=args['lr'])

    for epoch in range(1, args['epoch']+1):
        train(net, train_loader, args['criterion'], optimizer, args, epoch)
        val_acc = val(net, val_loader)
        
        if abs(val_acc - args['gt_val_acc']) <= 1e-5:
            torch.save(net, 'saved_models/model.pth')
            break
    
def train(net, train_loader, criterion, optimizer, args, epoch):
    my_shape = args['input_shape']
    sticker = read_im('../smiley2.png', 7, 7)
    sticker_tensor = img_to_tensor(sticker)
    sticker_tensor.requires_grad = False
    sticker_tensor = torch.mean(sticker_tensor, dim=1) # remove RGB
    sticker_tensor = tensor_normalize(sticker_tensor)
    gradcam_target = sticker_tensor.repeat(args['train_batch_size'], 1, 1) # batch
    if args['cuda']:
        gradcam_target = gradcam_target.cuda()

    net.train()
    nb = 0
    Acc_v = 0
    class_loss = AverageMeter()
    
    for i, data in enumerate(train_loader):
        X, Y = data  # X1 batchsize x 1 x 16 x 16
        X = Variable(X)
        Y = Variable(Y)
        if args['cuda']:
            X = X.cuda()
            Y = Y.cuda()
        N = len(X)
        nb = nb+N
        
        outputs = net(X)
        Acc_v = Acc_v + (outputs.argmax(1) - Y).nonzero().size(0)

        optimizer.zero_grad()

        # normal training
        if criterion == 1:
            loss_fcn = torch.nn.CrossEntropyLoss()
            loss = loss_fcn(outputs, Y)
        # bullshit loss to illustrate double-gradient
        if criterion == 3:
            my_output = outputs[0]
            dydw = torch.autograd.grad(my_output[0], net.my_model.classifier.parameters(), create_graph=True)
            loss = torch.sum(torch.abs(dydw[0]))
        # gradcam loss
        if criterion == 2:
            gradcam = differentiable_cam(model=net, input=X, cuda=args['cuda'])
            loss = torch.sum(torch.abs(gradcam[0] - gradcam_target))/gradcam_target.shape[0]/gradcam_target.shape[1]/gradcam_target.shape[2]

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

        if args['cuda']:
            X = X.cuda()
            Y = Y.cuda()

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
    args = vars(args)
    for key in args.keys():
        hps[key] = args[key]

    main(hps)
