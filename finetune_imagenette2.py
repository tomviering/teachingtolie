#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 12:53:02 2019

@author: ziqi
"""
import argparse
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.autograd import Variable
from dataset import load_cifar, Imagenette
# from network import VGG_exp1, VGG_exp2
from explanation import differentiable_cam
from network import VGG_final, Alexnet_final
from utils import AverageMeter, mkdir, build_gradcam_target, val_vis_batch, loss_gradcam, print_progress
from loss import gradcam_loss
#%%
hps = {
    'nb_classes': 2,
    'train_batch_size': 10,
    'val_batch_size': 10,
    'epoch': 5,
    'lr': 1e-3,
    'weight_decay': 2e-4,
    'input_shape': (224, 224),
    'test_domain': 10,
    'print_freq': 10,
    'gt_val_acc': 0.78,
    'criterion': 2,
    'loss': 2,
    'dataset': 'cifar',
    'network': 'vgg',
    'alpha_c': 1,
    'alpha_g': 1,
    'vis_name': 'temp',
    'optimizer': 'adam'
}


#%%
def main():

    mkdir('saved_models/')
    # load dataset
    trainset = Imagenette(mode='train', input_shape=hps['input_shape'])
    valset = Imagenette(mode='val', input_shape=hps['input_shape'])
    hps['nb_classes'] = 10

    # define network
    if hps['network'] == 'vgg':
        net = VGG_final()
        hps['gradcam_shape'] = (7, 7)
    if hps['network'] == 'alexnet':
        net = Alexnet_final()
        hps['gradcam_shape'] = (6, 6)

    net.my_model.classifier[6] = torch.nn.Linear(4096, hps['nb_classes'], bias=True)

    if hps['cuda']:
        net = net.cuda()

    train_loader = DataLoader(trainset, batch_size=hps['train_batch_size'], shuffle=True, num_workers=1)
    val_loader = DataLoader(valset, batch_size=hps['val_batch_size'], shuffle=False, num_workers=1)

    # define loss function
    criterion = gradcam_loss(hps['alpha_c'], 0)
    target_parameters = net.my_model.classifier[6].parameters()

    if hps['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(target_parameters, lr=hps['lr'])
    if hps['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(target_parameters, lr=hps['lr'])

    mkdir('vis/%s/' % hps['vis_name'])
    gradcam_target = torch.zeros((0, 0))
    #gt_val_acc, _ = val(net, val_loader, criterion, gradcam_target, do_gradcam=False)
    #print('validation accuracy before finetuning: %.5f' % gt_val_acc)
    
#%%    
    for epoch in range(1, hps['epoch'] + 1):

        start = time.time()
        train(net, train_loader, criterion, optimizer, epoch, gradcam_target, do_gradcam=False)
        end = time.time()
        print('epoch took %d seconds' % (end - start))
        print('epoch took approximately %d minutes' % np.floor((end - start) / 60))

        #val_vis_batch(net, val_loader, num=5, save=True, fn='vis/%s/epoch%d_' % (hps['vis_name'], epoch), cuda=hps['cuda'])
        (val_acc, l_g) = val(net, val_loader, criterion, gradcam_target, do_gradcam=False)

    print('model trained until completion! saving...')
    torch.save(net.my_model.classifier[6], 'saved_models/classifier6_imagenette2.pth')

def train(net, train_loader, criterion, optimizer, epoch, gradcam_target, do_gradcam=False):
    net.train()
    nb = 0
    Acc_v = 0
    meter_a = AverageMeter()
    meter_c = AverageMeter()
    meter_g = AverageMeter()
    meter_t = AverageMeter()

    for i, data in enumerate(train_loader):
        start = time.time()

        X, Y = data  # X1 batchsize x 1 x 16 x 16
        X = Variable(X)
        Y = Variable(Y)
        if hps['cuda']:
            X = X.cuda()
            Y = Y.cuda()
        N = len(X)
        nb = nb + N

        output, features = net(X)
        Acc_v = Acc_v + (output.argmax(1) - Y).nonzero().size(0)

        optimizer.zero_grad()

        if not do_gradcam:
            exp = torch.zeros_like(features[:, 0, :, :])
            gradcam_target = torch.zeros_like(features[0, 0, :, :])
        else:
            exp, _ = differentiable_cam(net, X, cuda=hps['cuda'])

        loss = criterion(exp, output, gradcam_target.repeat(exp.size()[0], 1, 1), Y)
        loss[0].backward()
        optimizer.step()

        meter_a.update(loss[0].data.item(), N)
        meter_c.update(loss[1].data.item(), N)
        meter_g.update(loss[2].data.item(), N)

        end = time.time()
        delta_t = (end - start)
        meter_t.update(delta_t, 1)
        time_per_it = meter_t.avg
        time_per_epoch = (len(train_loader) * time_per_it / 60)

        if i % hps['print_freq'] == 0:
            print('[epoch %d], [iter %d / %d], [all loss %.5f] [class loss %.5f] [gradcam loss %.5f ] [time per epoch (minutes) %.1f]'
                % (epoch, i + 1, len(train_loader), meter_a.avg, meter_c.avg, meter_g.avg, time_per_epoch))
        # print(val_acc)
    train_acc = (nb - Acc_v) / nb
    print("train acc: %.5f" % train_acc)


# %%
def val(net, val_loader, criterion, gradcam_target, do_gradcam = False):
    net.eval()
    Acc_v = 0
    nb = 0
    meter_g = AverageMeter()

    progress = -1

    for i, data in enumerate(val_loader):

        progress = print_progress(progress, i, len(val_loader))

        X, Y = data
        X = Variable(X)
        Y = Variable(Y)
        if hps['cuda']:
            X = X.cuda()
            Y = Y.cuda()

        N = len(X)
        nb = nb + len(X)

        output, features = net(X)
        Acc_v = Acc_v + (output.argmax(1) - Y).nonzero().size(0)

        if not do_gradcam:
            exp = torch.zeros_like(features[:, 0, :, :])
            gradcam_target = torch.zeros_like(features[0, 0, :, :])
        else:
            exp, _ = differentiable_cam(net, X, cuda=hps['cuda'])

        loss = criterion(exp, output, gradcam_target.repeat(exp.size()[0], 1, 1), Y)
        meter_g.update(loss[2].data.item(), N)

    val_acc = (nb - Acc_v) / nb

    print("val acc: %.5f" % val_acc)
    print('gradcam loss %.5f' % meter_g.avg)
    return (val_acc, meter_g.avg)




#%%
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=bool, default=False)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--criterion', type=int, default=3)
    parser.add_argument('--lambda', type=float, default=1e-2)
    parser.add_argument('--vis_name', default='test')
    parser.add_argument('--optimizer', default='adam')
    args = parser.parse_args()
    return args


#%%
if __name__ == '__main__':

    args = get_args()
    args = vars(args)
    for key in args.keys():
        hps[key] = args[key]

    print('hyperparameter settings:',hps)

    main()
