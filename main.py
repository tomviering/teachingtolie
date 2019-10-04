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
    'lr': 1e-2,
    'weight_decay': 2e-4,
    'input_shape': (224, 224),
    'test_domain': 10,
    'print_freq': 1,
    'gt_val_acc': 0.78,
    'criterion': 2,
    'loss': 2,
    'dataset': 'imagenette',
    'lambda': 0.0
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
    net = VGG_final()

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

    val_acc = val(net, val_loader, cuda=hps['cuda'])

    # define loss function
    optimizer = torch.optim.Adam(net.my_model.parameters(), lr=hps['lr'])

    mkdir('vis/')
    val_vis_batch(net, val_loader, num=5, save=True, fn='vis/epoch0_', cuda=hps['cuda'])

    for epoch in range(1, hps['epoch'] + 1):

        start = time.time()
        train(net, train_loader, hps['criterion'], optimizer, epoch)
        end = time.time()
        print('epoch took %d seconds' % (end - start))
        print('epoch took approximately %d minutes' % np.floor((end - start) / 60))

        val_vis_batch(net, val_loader, num=5, save=True, fn='vis/epoch%d_' % epoch, cuda=hps['cuda'])
        val_acc = val(net, val_loader, cuda=hps['cuda'])

        if hps['criterion'] == 1 and val_acc == 1.0:
            print('model trained until completion! saving...')
            torch.save(net.my_model.classifier[6], 'saved_models/classifier6.pth')
            break

        if abs(val_acc - hps['gt_val_acc']) <= 1e-5:
            torch.save(net, 'saved_models/model.pth')
            break



def train(net, train_loader, criterion, optimizer, epoch):
    my_shape = hps['input_shape']
    sticker = read_im('smiley2.png', 7, 7)
    sticker_tensor = img_to_tensor(sticker)
    sticker_tensor.requires_grad = False
    sticker_tensor = torch.mean(sticker_tensor, dim=1)  # remove RGB
    sticker_tensor = tensor_normalize(sticker_tensor)
    gradcam_target = sticker_tensor.repeat(hps['train_batch_size'], 1, 1)  # batch
    if hps['cuda']:
        gradcam_target = gradcam_target.cuda()

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
        # print(outputs.shape)
        # print(Y.shape)
        # print(Y)
        Acc_v = Acc_v + (outputs.argmax(1) - Y).nonzero().size(0)

        optimizer.zero_grad()

        loss_fcn = torch.nn.CrossEntropyLoss()
        # normal training
        if criterion == 1:  # this loss doesnt make any sense since output and Y are not same size...
            loss = loss_fcn(outputs, Y)
        # gradcam loss
        if criterion == 2:
            loss = gradcam_loss(X, net, gradcam_target, cuda=hps['cuda'])
        if criterion == 3:
            loss = loss_fcn(outputs, Y) #+ hps['lambda']*gradcam_loss(X, net, gradcam_target)

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
    parser.add_argument('--criterion', type=int, default=2)
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
