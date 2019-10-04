#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 12:53:02 2019

@author: ziqi
"""
import argparse
import time

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.autograd import Variable
from dataset import load_cifar, Imagenette
# from network import VGG_exp1, VGG_exp2
from explanation import differentiable_cam
from network import VGG_final, Alexnet_final
from utils import *

hps = {
    'nb_classes': 2,
    'train_batch_size': 8,
    'val_batch_size': 3,
    'epoch': 500,
    'lr': 1e-3,
    'weight_decay': 2e-4,
    'input_shape': (224, 224),
    'test_domain': 10,
    'print_freq': 1,
    'gt_val_acc': 0.78,
    'criterion': 2,
    'loss': 2,
    'dataset': 'imagenette',
    'lambda': 0.01,
    'vis_name': 'temp',
    'optimizer': 'adam'
}


def main():


    mkdir('saved_models/')

    if hps['dataset'] == 'imagenette':
        trainset = Imagenette(mode='val', input_shape=hps['input_shape'])
        valset = Imagenette(mode='val', input_shape=hps['input_shape'])
        hps['nb_classes'] = 10

    if hps['dataset'] == 'cifar':
        trainset = load_cifar(mode='val', input_shape=hps['input_shape'])
        valset = load_cifar(mode='val', input_shape=hps['input_shape'])
        hps['nb_classes'] = 10

    # define network
    network = 'vgg'
    if network == 'vgg':
        net = VGG_final()
        hps['gradcam_shape'] = (7, 7)
    if network == 'alexnet':
        net = Alexnet_final()
        hps['gradcam_shape'] = (6, 6)

    if hps['dataset'] == 'imagenette':
        print('loading pretrained model for imagenette...')
        net.my_model.classifier[6] = torch.load('saved_models/classifier6_imagenette.pth')
        # this model achieves 100% validation accuracy
    else:
        net.my_model.classifier[6] = torch.nn.Linear(4096, hps['nb_classes'], bias=True)


    if hps['cuda']:
        net = net.cuda()

    train_loader = DataLoader(trainset, batch_size=hps['train_batch_size'], shuffle=False, num_workers=1)
    val_loader = DataLoader(valset, batch_size=hps['val_batch_size'], shuffle=False, num_workers=1)

    val(net, val_loader, hps)

    # define loss function
    target_parameters = net.my_model.parameters()

    if hps['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(target_parameters, lr=hps['lr'])
    if hps['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(target_parameters, lr=hps['lr'])

    mkdir('vis/%s/' % hps['vis_name'])
    val_vis_batch(net, val_loader, num=5, save=True, fn='vis/%s/epoch0_' % hps['vis_name'], cuda=hps['cuda'])

    for epoch in range(1, hps['epoch'] + 1):

        start = time.time()
        train(net, train_loader, hps['criterion'], optimizer, epoch)
        end = time.time()
        print('epoch took %d seconds' % (end - start))
        print('epoch took approximately %d minutes' % np.floor((end - start) / 60))

        val_vis_batch(net, val_loader, num=5, save=True, fn='vis/%s/epoch%d_' % (hps['vis_name'], epoch), cuda=hps['cuda'])
        (val_acc, val_c, val_g) = val(net, val_loader, hps)

        if hps['criterion'] == 1 and val_acc == 1.0:
            print('model trained until completion! saving...')
            torch.save(net.my_model.classifier[6], 'saved_models/classifier6.pth')
            break

def train(net, train_loader, criterion, optimizer, epoch):

    gradcam_target = build_gradcam_target(gradcam_shape=hps['gradcam_shape'], cuda=hps['cuda'], batch_size=hps['train_batch_size'])

    net.train()
    nb = 0
    Acc_v = 0
    meter_c = AverageMeter()
    meter_g = AverageMeter()

    for i, data in enumerate(train_loader):
        X, Y = data  # X1 batchsize x 1 x 16 x 16
        X = Variable(X)
        Y = Variable(Y)
        if hps['cuda']:
            X = X.cuda()
            Y = Y.cuda()
        N = len(X)
        nb = nb + N

        outputs = net(X)
        # print(outputs.shape)
        # print(Y.shape)
        # print(Y)
        Acc_v = Acc_v + (outputs.argmax(1) - Y).nonzero().size(0)

        optimizer.zero_grad()

        loss_fcn = torch.nn.CrossEntropyLoss()

        loss_c = torch.zeros(1)
        loss_g = torch.zeros(1)

        # normal training
        if criterion == 1:  # this loss doesnt make any sense since output and Y are not same size...
            loss_c = loss_fcn(outputs, Y)
            loss = loss_c
        # gradcam loss
        if criterion == 2:
            loss_g = loss_gradcam(X, net, gradcam_target, cuda=hps['cuda'])
            loss = loss_g
        if criterion == 3:
            loss_c = loss_fcn(outputs, Y)
            loss_g = loss_gradcam(X, net, gradcam_target, cuda=hps['cuda'])
            loss = loss_c + hps['lambda']*loss_g

        loss.backward()
        optimizer.step()

        meter_c.update(loss_c.data.item(), N)
        meter_g.update(loss_g.data.item(), N)

        if epoch % hps['print_freq'] == 0:
            print('[epoch %d], [iter %d / %d], [class loss %.5f] [gradcam loss %.5f ][memory used %d]'
                  % (epoch, i + 1, len(train_loader), meter_c.avg, meter_g.avg, get_gpu_memory_map()[0]))
        # print(val_acc)
    train_acc = (nb - Acc_v) / nb
    print("train acc: %.5f" % train_acc)


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


if __name__ == '__main__':

    args = get_args()
    args = vars(args)
    for key in args.keys():
        hps[key] = args[key]

    print('hyperparameter settings:')
    print(hps)

    main()
