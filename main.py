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
    'lambda': 0.01
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
    else:
        net.my_model.classifier[6] = torch.nn.Linear(4096, hps['nb_classes'], bias=True)


    if hps['cuda']:
        net = net.cuda()

    train_loader = DataLoader(trainset, batch_size=hps['train_batch_size'], shuffle=False, num_workers=1)
    val_loader = DataLoader(valset, batch_size=hps['val_batch_size'], shuffle=False, num_workers=1)

    val_acc = val(net, val_loader)

    # define loss function
    optimizer = torch.optim.Adam(net.my_model.parameters(), lr=hps['lr'])

    mkdir('vis/')
    val_vis_batch(net, val_loader, num=5, save=True, fn='vis/epoch0_')

    for epoch in range(1, hps['epoch'] + 1):

        start = time.time()
        train(net, train_loader, hps['criterion'], optimizer, epoch)
        end = time.time()
        print('epoch took %d seconds' % (end - start))
        print('epoch took approximately %d minutes' % np.floor((end - start) / 60))

        val_vis_batch(net, val_loader, num=5, save=True, fn='vis/epoch%d_' % epoch)
        val_acc = val(net, val_loader)

        if hps['criterion'] == 1 and val_acc == 1.0:
            print('model trained until completion! saving...')
            torch.save(net.my_model.classifier[6], 'saved_models/classifier6.pth')
            break

        if abs(val_acc - hps['gt_val_acc']) <= 1e-5:
            torch.save(net, 'saved_models/model.pth')
            break


def save_im(X, cam, output, Y, fn='', save=False):
    print('got %d images' % X.shape[0])

    for i in range(0, X.shape[0]):

        plt.figure(i * 3 + 0)
        tensor_plot(X[i, :, :, :])
        plt.axis('off')
        if save:
            plt.savefig(fn + str(i) + 'im_im.png')
            plt.close()

        plt.figure(i * 3 + 1)
        pic = show_cam_on_tensor(X[i, :, :, :], cam[i, :, :])
        plt.imshow(pic)
        plt.axis('off')
        if save:
            plt.savefig(fn + str(i) + 'im_overlay.png')
            plt.close()

        plt.figure(i * 3 + 2)
        pic = show_cam_on_tensor(X[i, :, :, :] * 0.0, cam[i, :, :])
        plt.imshow(pic)
        plt.axis('off')
        if save:
            plt.savefig(fn + str(i) + 'im_gradcam.png')
            plt.close()

        print('showing image %d of %d' % (i, X.shape[0]))
        print_predictions(output[i, :].squeeze(), 5)
        print('ground truth %d' % (Y[i]))
        print('close figures to continue...')

        if save == False:
            plt.show()


def gradcam_loss(X, net, gradcam_target):
    gradcam = differentiable_cam(model=net, input=X, cuda=hps['cuda'])
    if torch.sum(torch.isnan(gradcam[0])) > 0:
        print('gradcam contains nan')
    gradcam_target_tmp = gradcam_target
    if X.shape[0] != gradcam_target.shape[0]:
        gradcam_target_tmp = gradcam_target[0:X.shape[0], :, :]
    num = gradcam_target_tmp.shape[0] * gradcam_target_tmp.shape[1] * gradcam_target_tmp.shape[2]
    diff = gradcam[0] - gradcam_target_tmp
    if hps['loss'] == 1:
        diff = torch.abs(diff)
    if hps['loss'] == 2:
        diff = torch.mul(diff, diff)
    loss = torch.sum(diff) / num
    return loss


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
            # how does it even work???
            loss = loss_fcn(outputs, Y)
        # gradcam loss
        if criterion == 2:
            loss = gradcam_loss(X, net, gradcam_target)
        if criterion == 3:
            loss = loss_fcn(outputs, Y) + hps['lambda']*gradcam_loss(X, net, gradcam_target)

        loss.backward()
        optimizer.step()

        class_loss.update(loss.data.item(), N)

        if epoch % hps['print_freq'] == 0:
            print('[epoch %d], [iter %d / %d], [class loss %.5f] [memory used %d]'
                  % (epoch, i + 1, len(train_loader), class_loss.avg, get_gpu_memory_map()[0]))
        # print(val_acc)
    train_acc = (nb - Acc_v) / nb
    print("train acc: %.5f" % train_acc)


def val_vis_batch(net, val_loader, num=hps['val_batch_size'], save=False, fn=''):
    net.eval()

    in_shape = hps['input_shape']
    X_total = torch.zeros(0, 3, in_shape[0], in_shape[1])
    cam_total = torch.zeros(0, 0)
    Y_total = torch.zeros(0, 0)
    output_total = torch.zeros(0, 0)
    nb = 0
    first = True

    for i, data in enumerate(val_loader):
        X, Y = data
        X = Variable(X)
        Y = Variable(Y)

        if hps['cuda']:
            X = X.cuda()
            Y = Y.cuda()

        (cam, output) = differentiable_cam(model=net, input=X, cuda=hps['cuda'])

        if first:
            first = False
            X_total = torch.zeros(0, X.shape[1], X.shape[2], X.shape[3])
            Y_total = torch.zeros(0, dtype=Y.dtype)
            output_total = torch.zeros(0, output.shape[1])
            cam_total = torch.zeros(0, cam.shape[1], cam.shape[2])

        X_total = torch.cat((X_total, X.cpu()), dim=0)
        Y_total = torch.cat((Y_total, Y.cpu()), dim=0)
        output_total = torch.cat((output_total, output.cpu()), dim=0)
        cam_total = torch.cat((cam_total, cam.cpu()), dim=0)

        nb = nb + X.shape[0]

        if nb > num:
            break

    save_im(X_total[0:num, :, :, :], cam_total[0:num, :, :], output_total, Y_total, save=save, fn=fn)


def val(net, val_loader):
    net.eval()
    Acc_v = 0
    nb = 0
    print('computing accuracy on validation data...')
    percentage = -1
    for i, data in enumerate(val_loader):

        percentage_tmp = np.floor(i / len(val_loader) * 10) * 10
        if percentage_tmp > percentage:
            percentage = percentage_tmp
            print('percentage %d of %d' % (percentage, 100))

        X, Y = data
        X = Variable(X)
        Y = Variable(Y)

        if hps['cuda']:
            X = X.cuda()
            Y = Y.cuda()

        nb = nb + len(X)

        outputs = net(X)
        Acc_v = Acc_v + (outputs.argmax(1) - Y).nonzero().size(0)

    val_acc = (nb - Acc_v) / nb

    print("val acc: %.5f" % val_acc)
    return val_acc


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
