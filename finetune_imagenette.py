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
from network import VGG_final
from utils import *

hps = {
    'nb_classes': 2,
    'train_batch_size': 8,
    'val_batch_size': 3,
    'epoch': 500,
    'lr': 1e-3,
    'weight_decay': 2e-4,
    'input_shape': (224, 224),
    'print_freq': 1
}


def main():

    mkdir('saved_models/')

    trainset = Imagenette(mode='val', input_shape=hps['input_shape'])
    valset = Imagenette(mode='val', input_shape=hps['input_shape'])
    hps['nb_classes'] = 10

    # define network
    net = VGG_final()

    net.my_model.classifier[6] = torch.nn.Linear(4096, hps['nb_classes'], bias=True)

    if hps['cuda']:
        net = net.cuda()

    train_loader = DataLoader(trainset, batch_size=hps['train_batch_size'], shuffle=False, num_workers=1)
    val_loader = DataLoader(valset, batch_size=hps['val_batch_size'], shuffle=False, num_workers=1)

    optimizer = torch.optim.Adam(net.my_model.classifier[6].parameters(), lr=hps['lr'])

    for epoch in range(1, hps['epoch'] + 1):

        start = time.time()
        train(net, train_loader, optimizer, epoch)
        end = time.time()
        print('epoch took %d seconds' % (end - start))
        print('epoch took approximately %d minutes' % np.floor((end - start) / 60))

        val_acc = val(net, val_loader, cuda=hps['cuda'])

        if val_acc == 1.0:
            print('model trained until completion! saving...')
            torch.save(net.my_model.classifier[6], 'saved_models/classifier6_imagenette.pth')
            break

def train(net, train_loader, optimizer, epoch):
    my_shape = hps['input_shape']

    net.train()
    nb = 0
    Acc_v = 0
    class_loss = AverageMeter()

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
        Acc_v = Acc_v + (outputs.argmax(1) - Y).nonzero().size(0)

        optimizer.zero_grad()

        loss_fcn = torch.nn.CrossEntropyLoss()
        loss = loss_fcn(outputs, Y)

        loss.backward()
        optimizer.step()

        class_loss.update(loss.data.item(), N)

        if epoch % hps['print_freq'] == 0:
            print('[epoch %d], [iter %d / %d], [class loss %.5f] [memory used %d]'
                  % (epoch, i + 1, len(train_loader), class_loss.avg, get_gpu_memory_map()[0]))
        # print(val_acc)
    train_acc = (nb - Acc_v) / nb
    print("train acc: %.5f" % train_acc)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=bool, default=False)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
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
