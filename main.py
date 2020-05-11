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
from network import VGG_final, Alexnet_final
from utils import AverageMeter, mkdir, val_vis_batch, print_progress, \
    get_gpu_memory_map
from sticker import prepare_batch, build_gradcam_target_sticker, build_gradcam_target_constant
from loss import random_loss, local_constant_loss
from earlystop import EarlyStopping
from explanation import differentiable_cam
from utils import read_im, rescale_batch

#%%
hps = {
    'nb_classes': -1, # will be determined by dataset
    'input_shape': (224,224)
}

# returns the sticker in the shape [width x height] (greyscale)
def get_sticker_tensor(filename, width, height):
    sticker_tensor = read_im(filename, width, height)
    sticker_tensor.requires_grad = False
    sticker_tensor = torch.mean(sticker_tensor, dim=1)  # remove RGB
    return rescale_batch(sticker_tensor)

#%%
def main():
    mkdir('saved_models/')
    # load dataset
    if hps['dataset'] == 'imagenette':
        trainset = Imagenette(mode='train', input_shape=hps['input_shape'])
        valset = Imagenette(mode='val', input_shape=hps['input_shape'])
        hps['nb_classes'] = 10

    if hps['dataset'] == 'cifar':
        trainset = load_cifar(mode='train', input_shape=hps['input_shape'])
        valset = load_cifar(mode='val', input_shape=hps['input_shape'])
        hps['nb_classes'] = 10

    # define network
    if hps['network'] == 'vgg':
        net = VGG_final(pretrained=hps['pretrained'])
        hps['gradcam_shape'] = (14, 14)
    if hps['network'] == 'alexnet':
        net = Alexnet_final(pretrained=hps['pretrained'])
        hps['gradcam_shape'] = (13, 13)

    if not hps['pretrained']:
        net.my_model.classifier[6] = torch.nn.Linear(4096, hps['nb_classes'], bias=True)
    else:
        if hps['dataset'] == 'imagenette':
            print('loading pretrained model for imagenette...')
            if hps['cuda']:
                net.my_model.classifier[6] = torch.load('saved_models/classifier6_imagenette2.pth')
            else:
                net.my_model.classifier[6] = torch.load('saved_models/classifier6_imagenette2.pth',
                                                        map_location=torch.device('cpu'))
            # this model achieves 97.4% validation accuracy on imagenette validation set (trained _only_ for 5 epochs)
        else:
            net.my_model.classifier[6] = torch.nn.Linear(4096, hps['nb_classes'], bias=True)

    if hps['cuda']:
        net = net.cuda()

    train_loader = DataLoader(trainset, batch_size=hps['train_batch_size'], shuffle=True, num_workers=hps['num_workers'], pin_memory=True)
    val_loader = DataLoader(valset, batch_size=hps['val_batch_size'], shuffle=False, num_workers=hps['num_workers'], pin_memory=True)

    # input, network, output, label
    # define loss function
    if (hps['attack_type'] == 'constant'):
        criterion = local_constant_loss(hps['lambda_c'], hps['lambda_g'], hps['lambda_a'])
    if (hps['attack_type'] == 'random'):
        criterion = random_loss(hps['lambda_c'], hps['lambda_g'])

    target_parameters = net.my_model.parameters()

    if hps['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(target_parameters, lr=hps['lr'])
    if hps['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(target_parameters, lr=hps['lr'])

    
    # early stop
    early_stopping = EarlyStopping(patience=hps['patience'], verbose=True, vis_name=hps['vis_name'])
        
    mkdir('vis/%s/' % hps['vis_name'])
    val_vis_batch(net, val_loader, num=5, save=True, fn='vis/%s/epoch0_' % hps['vis_name'], cuda=hps['cuda'])

    gradcam_shape = hps['gradcam_shape']

    if hps['attack_type'] == 'backdoor':
        # this is for the backdoor
        sticker_backdoor = get_sticker_tensor('smiley2.png', 14, 14)
        gradcam_target_builder = build_gradcam_target_sticker(sticker_backdoor, gradcam_shape)
    else:
        # this is for the sticker constant
        sticker_constant = get_sticker_tensor('smiley2.png', gradcam_shape[0], gradcam_shape[1])
        gradcam_target_builder = build_gradcam_target_constant(sticker_constant)

    if hps['attack_type'] != "random":
        hps['index_attack'] = find_least_important_alpha(net, train_loader, optimizer)

    gt_val_acc, _, _ = val(net, val_loader, criterion, gradcam_target_builder)
    print('validation accuracy before finetuning: %.5f' % gt_val_acc)
    
#%%    
    for epoch in range(1, hps['epoch'] + 1):

        start = time.time()
        train(net, train_loader, criterion, optimizer, epoch, gradcam_target_builder)
        end = time.time()
        print('epoch took %d seconds' % (end - start))
        print('epoch took %.1f minutes' % ((end - start) / 60))

        val_vis_batch(net, val_loader, num=5, save=True, fn='vis/%s/epoch%d_' % (hps['vis_name'], epoch), cuda=hps['cuda'])
        (val_acc, l_g, l_a) = val(net, val_loader, criterion, gradcam_target_builder)
        
        early_stopping(l_a, net)        
        if early_stopping.early_stop:
            print("Early stopping")
            break

def find_least_important_alpha(net, train_loader, optimizer):
    net.eval()
    nb = 0

    epoch = -1
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

        optimizer.zero_grad()

        batchsize = X.shape[0]

        exp, _, alpha, _ = differentiable_cam(net, X, cuda=hps['cuda'])
        # alpha is shape: [batchsize x channels]

        my_alpha = alpha.detach().cpu().numpy()
        if i == 0:
            all_alphas = my_alpha
        else:
            all_alphas = np.append(all_alphas, my_alpha, 0)

        alpha_summed = torch.sum(torch.abs(alpha.detach()), 0)

        if i == 0:
            alpha_maxed = torch.zeros_like(alpha_summed)

        alpha_maxed = torch.max(torch.max(alpha.detach(), 0).values, alpha_maxed)

        if i == 0:
            alpha_total = torch.zeros_like(alpha_summed)

        alpha_total = alpha_total + alpha_summed

        end = time.time()
        delta_t = (end - start)
        meter_t.update(delta_t, 1)
        time_per_it = meter_t.avg
        time_per_epoch = (len(train_loader) * time_per_it / 60)

        if i % hps['print_freq'] == 0:
            print('[epoch %d], [iter %d / %d], [time per epoch (minutes) %.1f] [memory %d MB]'
                % (epoch, i + 1, len(train_loader), time_per_epoch, get_gpu_memory_map(hps['cuda'])))
        # print(val_acc)

    print('done, cumulative absolute value of the alphas is given below')
    print(alpha_total/nb)

    print('maximum alpha over the whole training set given below')
    print(alpha_maxed)

    print('maximum over all alpha')
    print(torch.max(alpha_maxed).values)

    best_alpha = torch.argmin(alpha_total)
    print('best alpha is %d with value %f' % (best_alpha, alpha_total[best_alpha]))

    print('saving all alpha''s of the train set to alphas.txt...')
    np.savetxt('alphas.txt', all_alphas)
    return best_alpha

#%%
def train(net, train_loader, criterion, optimizer, epoch, gradcam_target_builder):
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
        if hps['attack_type'] == 'backdoor':
            X = prepare_batch(X)      
        gradcam_target = gradcam_target_builder.forward(X)
        X = Variable(X)
        Y = Variable(Y)
        if hps['cuda']:
            X = X.cuda()
            Y = Y.cuda()
            gradcam_target = gradcam_target.cuda()
        N = len(X)
        nb = nb + N

        output, features = net(X)
        Acc_v = Acc_v + (output.argmax(1) - Y).nonzero().size(0)

        optimizer.zero_grad()

        batchsize = X.shape[0]

        criterion_args = {
            'X': X,
            'net': net,
            'output': output,
            'Y': Y,
            'gradcam_target': gradcam_target.repeat(batchsize, 1, 1),
            'cuda': hps['cuda'],
            'index_attack': hps['index_attack']
        }

        loss = criterion(criterion_args)

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
            print('[epoch %d], [iter %d / %d], [all loss %.5f] [class loss %.5f] [gradcam loss %.5f ] [time per epoch (minutes) %.1f] [memory %d MB]'
                % (epoch, i + 1, len(train_loader), meter_a.avg, meter_c.avg, meter_g.avg, time_per_epoch, get_gpu_memory_map(hps['cuda'])))
        # print(val_acc)
    train_acc = (nb - Acc_v) / nb
    print("train acc: %.5f" % train_acc)
    
#%%    
def val(net, val_loader, criterion, gradcam_target_builder):
    net.eval()
    Acc_v = 0
    nb = 0
    meter_g = AverageMeter()
    meter_a = AverageMeter()
    meter_c = AverageMeter()

    progress = -1

    for i, data in enumerate(val_loader):

        progress = print_progress(progress, i, len(val_loader))

        X, Y = data
        
        if hps['attack_type'] == 'backdoor':
            X = prepare_batch(X)      
        gradcam_target = gradcam_target_builder.forward(X)
        X = Variable(X)
        Y = Variable(Y)
        if hps['cuda']:
            X = X.cuda()
            Y = Y.cuda()
            gradcam_target = gradcam_target.cuda()    
        N = len(X)
        nb = nb + len(X)
        
        output, features = net(X)
        Acc_v = Acc_v + (output.argmax(1) - Y).nonzero().size(0)

        batchsize = X.shape[0]

        criterion_args = {
            'X': X,
            'net': net,
            'output': output,
            'Y': Y,
            'gradcam_target': gradcam_target.repeat(batchsize, 1, 1),
            'cuda': hps['cuda'],
            'index_attack': hps['index_attack']
        }

        loss = criterion(criterion_args)
        meter_g.update(loss[2].data.item(), N)
        meter_a.update(loss[0].data.item(), N)
        meter_c.update(loss[1].data.item(), N)

    val_acc = (nb - Acc_v) / nb

    print("val acc: %.5f" % val_acc)
    print('gradcam loss %.5f' % meter_g.avg)
    print("val loss: %.5f" % meter_c.avg)
    return (val_acc, meter_g.avg, meter_a.avg)

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
#%%
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=False, type=str2bool)
    parser.add_argument('--train_batch_size', type=int, default=10)
    parser.add_argument('--val_batch_size', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--vis_name', default='test')
    parser.add_argument('--optimizer', default='adam', choices=['sgd', 'adam']) # sgd or adam
    parser.add_argument('--lambda_c', default=1.0, type=float)
    parser.add_argument('--lambda_g', default=1.0, type=float)
    parser.add_argument('--lambda_a', default=1.0, type=float)
    parser.add_argument('--dataset', default='imagenette', choices=['imagenette', 'cifar'])
    parser.add_argument('--network', default='vgg', choices=['vgg', 'alexnet'])
    parser.add_argument('--print_freq', default=100, type=int)
    parser.add_argument('--patience', default=20, type=int)
    parser.add_argument('--epoch', default=500, type=int)
    parser.add_argument('--pretrained', default=True, type=str2bool)
    parser.add_argument('--RAM_dataset', default=False, type=str2bool)
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--attack_type', default='constant', choices=['random', 'constant', 'backdoor'])
    args = parser.parse_args()
    return args


#%%
if __name__ == '__main__':

    args = get_args()
    args = vars(args)
    for key in args.keys():
        hps[key] = args[key]

    print('\n\n')
    print('*'*30)
    print('hyperparameter settings:')
    for key in hps.keys():
        print('%s: %s' % (key, str(hps[key])))
    print('*' * 30)
    print('\n\n')
    main()
