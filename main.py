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
from dataset import load_cifar, Imagenette, precomputedDataset
# from network import VGG_exp1, VGG_exp2
from network import VGG_final, Alexnet_final
from utils import AverageMeter, mkdir, val_vis_batch, print_progress, \
    get_gpu_memory_map, check_precomputed_dataloader
from sticker import prepare_batch, build_gradcam_target_sticker, build_gradcam_target_constant, get_vectors
from loss import random_loss, local_constant_loss, local_constant2_loss, constant_loss, local_constant_negative_loss, \
    center_loss_fixed, exp_validation
from earlystop import EarlyStopping
from explanation import differentiable_cam
from utils import read_im, rescale_batch, read_im_transformed
import os.path

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


def get_sticker_tensor_transformed(filename, width, height):
    sticker_tensor = read_im_transformed(filename, width, height)
    sticker_tensor.requires_grad = False
    return sticker_tensor


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

    print('number of training samples is %d' % len(trainset))
    print('number of validation samples is %d' % len(valset))

#%%  define network
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

    shuffle_train = True
    if hps['attack_type'] == 'backdoor':
        shuffle_train = False  # dont shuffle, because then we will confuse the order of objects when doing the pre computing...

#%% load dataset        
    train_loader = DataLoader(trainset, batch_size=hps['train_batch_size'], shuffle=shuffle_train, num_workers=hps['num_workers'], pin_memory=True)
    val_loader = DataLoader(valset, batch_size=hps['val_batch_size'], shuffle=False, num_workers=hps['num_workers'], pin_memory=True)

    # input, network, output, label
 #%%  define loss function for different attacks
    if hps['attack_type'] == 'random':
        criterion = random_loss(hps['lambda_c'], hps['lambda_g'])
        hps['loss_type'] = 'random'
    else:
        if hps['loss_type'] == 'local_constant':
            criterion = local_constant_loss(hps['lambda_c'], hps['lambda_g'], hps['lambda_a'])
        elif hps['loss_type'] == 'local_constant2':
            criterion = local_constant2_loss(hps['lambda_c'], hps['lambda_g'], hps['lambda_a'])
        elif hps['loss_type'] == 'constant':
            criterion = constant_loss(hps['lambda_c'], hps['lambda_g'])
        elif hps['loss_type'] == 'local_constant_negative':
            criterion = local_constant_negative_loss(hps['lambda_c'], hps['lambda_g'], hps['lambda_a'])
        elif hps['loss_type'] == 'center_loss_fixed':
            criterion = center_loss_fixed(hps['lambda_c'], hps['lambda_g'])
            
            
#%% define optimizer
    target_parameters = net.my_model.parameters()
    if hps['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(target_parameters, lr=hps['lr'])
    if hps['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(target_parameters, lr=hps['lr'])
        

#%% early stop
    early_stopping = EarlyStopping(patience=hps['patience'], verbose=True, vis_name=hps['vis_name'])        
    mkdir('vis/%s/' % hps['vis_name'])
    val_vis_batch(net, val_loader, num=5, save=True, fn='vis/%s/epoch0_' % hps['vis_name'], cuda=hps['cuda'])
    

#%% build gradcam target
    gradcam_shape = hps['gradcam_shape']
    if hps['attack_type'] == 'backdoor':
        # this is for the backdoor
        sticker = get_sticker_tensor_transformed('smiley2.png', gradcam_shape[0], gradcam_shape[1])
        gradcam_target_builder = build_gradcam_target_sticker(sticker, gradcam_shape)
    if hps['attack_type'] != 'backdoor':
        # this is for the sticker constant
        sticker = get_sticker_tensor(hps['sticker_img'], gradcam_shape[0], gradcam_shape[1])
        gradcam_target_builder = build_gradcam_target_constant(sticker)


#%% find leaset important alpha, or skip
    if hps['attack_type'] == 'random' or hps['loss_type'] == 'constant':
        hps['skip_find_alpha'] = True

    if not hps['skip_find_alpha']:
        hps['index_attack'] = find_least_important_alpha(net, train_loader, optimizer)

#%% precompute dataloader for backdoor
    if hps['attack_type'] == 'backdoor':
        print('precomputing training data...')

        if hps['minitrn']: # use subset of training for debugging purposes
            trainset_precomputed = precompute_stickers(net, train_loader, gradcam_target_builder, sticker, trainset, hps, 'trn_mini_precomputed', subset=True, subset_size=100)
        else:
            trainset_precomputed = precompute_stickers(net, train_loader, gradcam_target_builder, sticker, trainset, hps, 'trn_precomputed')
        print('precomputing validation data...')
        valset_precomputed = precompute_stickers(net, val_loader, gradcam_target_builder, sticker, valset, hps, 'val_precomputed')

        train_loader = DataLoader(trainset_precomputed, batch_size=hps['train_batch_size'], shuffle=True,
                                  num_workers=hps['num_workers'], pin_memory=True)
        val_loader = DataLoader(valset_precomputed, batch_size=hps['val_batch_size'], shuffle=False,
                                num_workers=hps['num_workers'], pin_memory=True)
        # check_precomputed_dataloader(val_loader)
        # these loaders return 5 arguments:
        # X
        # Y
        # X_corrupted_precomputed [image with stickers]
        # gradcam_target_precomputed [gradcam target tensor]
        # explenation_precomputed [explenation of the pretrained model - note, need to set PRETRAINED to TRUE !!!]
        if not hps['pretrained']:
            raise Exception('You need to use pretrained model for backdoor...')
    

#%% training loop
    print(hps)
    if not hps['skip_validation']:
        gt_val_acc, _, _ = val(net, val_loader, criterion, gradcam_target_builder, sticker)
        print('validation accuracy before finetuning: %.5f' % gt_val_acc)
        
    for epoch in range(1, hps['epoch'] + 1):

        print('*' * 25)
        print('STARTING TRAIN PHASE')
        print('*' * 25)

        start = time.time()
        train(net, train_loader, criterion, optimizer, epoch, gradcam_target_builder, sticker)
        end = time.time()
        print('epoch took %d seconds' % (end - start))
        print('epoch took %.1f minutes' % ((end - start) / 60))

        val_vis_batch(net, val_loader, num=5, save=True, fn='vis/%s/epoch%d_' % (hps['vis_name'], epoch), cuda=hps['cuda'])
        (val_acc, l_g, l_a) = val(net, val_loader, criterion, gradcam_target_builder, sticker)
        
        early_stopping(l_a, net)        
        if early_stopping.early_stop:
            print("Early stopping")
            break


def precompute_stickers(net, loader, gradcam_target_builder, sticker, original_dataset, hps, fn, dosave = True, doload = True, subset = False, subset_size = 10):

    fn_sticker = '%s_sticker.pt' % fn
    fn_exp_target = '%s_exp_target.pt' % fn
    fn_exp_original = '%s_exp_original.pt' % fn

    if doload and os.path.isfile(fn_sticker) and os.path.isfile(fn_exp_target) and os.path.isfile(fn_exp_original):
        print('loading saved precomputed tensors...')

        X_corrupted_precomputed = torch.load(fn_sticker)
        gradcam_target_precomputed = torch.load(fn_exp_target)
        explenation_precomputed = torch.load(fn_exp_original)

    else:
        print('going to precompute tensors... could take a while.')
        N = len(original_dataset)



        progress = -1

        for i, data in enumerate(loader):

            if i == subset_size and subset:
                break

            #print('batch %d memory %d MB' % (i, get_gpu_memory_map(hps['cuda'])))

            progress = print_progress(progress, i, len(loader))

            X, Y = data

            X_tocorrupt = X.clone()
            X_corrupted = prepare_batch(X_tocorrupt, gradcam_target_builder, sticker)

            gradcam_target = gradcam_target_builder.forward(X_corrupted)
            gradcam_target_nograd = gradcam_target.detach()

            if hps['cuda']:
                X = X.cuda()
                Y = Y.cuda()

            exp, _, _, _ = differentiable_cam(net, X, cuda=hps['cuda'])
            exp_copy = exp.detach().cpu()

            if i == 0:
                bs = X.shape[0]  # batchsize
                if subset:
                    N = bs*subset_size

                X_corrupted_precomputed = X.new_empty((N, X.shape[1], X.shape[2], X.shape[3]), dtype=None, device=torch.device('cpu'))
                gradcam_target_precomputed = gradcam_target_nograd.new_empty((N, gradcam_target.shape[1], gradcam_target.shape[2]), dtype=None, device=torch.device('cpu'))
                explenation_precomputed = exp_copy.new_empty((N, exp.shape[1], exp.shape[2]), dtype=None, device=torch.device('cpu'))

            start_ind = bs*i
            end_ind = bs*(i+1)

            if X.shape[0] == bs: # not the last batch
                X_corrupted_precomputed[start_ind:end_ind,:,:,:] = X_corrupted[:,:,:,:]
                gradcam_target_precomputed[start_ind:end_ind,:,:] = gradcam_target_nograd[:,:,:]
                explenation_precomputed[start_ind:end_ind,:,:] = exp_copy[:,:,:]
            else: # this is the last batch
                X_corrupted_precomputed[start_ind:, :, :, :] = X_corrupted[:, :, :, :]
                gradcam_target_precomputed[start_ind:, :, :] = gradcam_target_nograd[:, :, :]
                explenation_precomputed[start_ind:, :, :] = exp_copy[:, :, :]

        if dosave:
            torch.save(X_corrupted_precomputed,fn_sticker)
            torch.save(gradcam_target_precomputed, fn_exp_target)
            torch.save(explenation_precomputed, fn_exp_original)

    if subset:
        new_dataset = precomputedDataset(original_dataset, X_corrupted_precomputed, gradcam_target_precomputed,
                                         explenation_precomputed, length_override=True, N=X_corrupted_precomputed.shape[0])
    else:
        new_dataset = precomputedDataset(original_dataset, X_corrupted_precomputed, gradcam_target_precomputed, explenation_precomputed)
    return new_dataset


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
def train(net, train_loader, criterion, optimizer, epoch, gradcam_target_builder, sticker):
    net.train()
    
    nb = 0
    Acc_v = 0
    meter_a = AverageMeter()
    meter_c = AverageMeter()
    meter_g = AverageMeter()
    meter_t = AverageMeter()
    meter_w = AverageMeter()
    meter_oa = AverageMeter()

    for i, data in enumerate(train_loader):
        start = time.time()

        if hps['attack_type'] == 'backdoor':
            X, Y, X_sticker, expl_target, expl_original = data
            X = torch.cat((X, X_sticker), 0)
            gradcam_target = torch.cat((expl_original, expl_target), 0)
            Y = torch.cat((Y, Y), 0)
        else:
            X, Y = data
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

        

        batchsize = X.shape[0]

        if hps['attack_type'] == 'backdoor':
            gradcam_target_batch = gradcam_target
        else:
            gradcam_target_batch = gradcam_target.repeat(batchsize, 1, 1)

        criterion_args = {
            'X': X,
            'net': net,
            'output': output,
            'Y': Y,
            'gradcam_target': gradcam_target_batch,
            'cuda': hps['cuda'],
            'index_attack': hps['index_attack']
        }

        loss = criterion(criterion_args)

        optimizer.zero_grad()
        loss[0].backward()
        optimizer.step()
        

        meter_a.update(loss[0].data.item(), N)
        meter_c.update(loss[1].data.item(), N)
        meter_g.update(loss[2].data.item(), N)
        if hps['loss_type'] == 'local_constant' or hps['loss_type'] == 'local_constant2' or hps['loss_type'] == 'local_constant_negative':
            meter_w.update(loss[3].data.item(), N)
            meter_oa.update(loss[4].data.item(), N)

        end = time.time()
        delta_t = (end - start)
        meter_t.update(delta_t, 1)
        time_per_it = meter_t.avg
        time_per_epoch = (len(train_loader) * time_per_it / 60)

        other_losses_string = ''
        if hps['loss_type'] == 'local_constant' or hps['loss_type'] == 'local_constant2'or hps['loss_type'] == 'local_constant_negative':
            other_losses_string = '[alpha/weight loss %.5f] [other alpha loss %.5f ]' % (meter_w.avg, meter_oa.avg)
        if i % hps['print_freq'] == 0:
            print('[epoch %d], [iter %d / %d], [all loss %.5f] [class loss %.5f] [gradcam loss %.5f ] %s [time per epoch (minutes) %.1f] [memory %d MB]'
                % (epoch, i + 1, len(train_loader), meter_a.avg, meter_c.avg, meter_g.avg, other_losses_string, time_per_epoch, get_gpu_memory_map(hps['cuda'])))

    train_acc = (nb - Acc_v) / nb
    print("train acc: %.5f" % train_acc)
   
    
#%%   
def val(net, val_loader, criterion, gradcam_target_builder, sticker):
    net.eval()
    Acc_v = 0
    nb = 0
    meter_g = AverageMeter()
    meter_a = AverageMeter()
    meter_c = AverageMeter()
    meter_exp_ori_1 = AverageMeter()
    meter_exp_ori_2 = AverageMeter()
    meter_exp_sticker_1 = AverageMeter()
    meter_exp_sticker_2 = AverageMeter()
    if hps['attack_type'] == 'backdoor':
        exp_loss = exp_validation()
    progress = -1

    for i, data in enumerate(val_loader):

        progress = print_progress(progress, i, len(val_loader))

        if hps['attack_type'] == 'backdoor':
            X, Y, X_sticker, expl_target, expl_original = data
            X = torch.cat((X, X_sticker), 0)
            gradcam_target = torch.cat((expl_original, expl_target), 0)
            Y = torch.cat((Y, Y), 0)
        else:
            X, Y = data
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

        if hps['attack_type'] == 'backdoor':
            gradcam_target_batch = gradcam_target
        else:
            gradcam_target_batch = gradcam_target.repeat(batchsize, 1, 1)

        criterion_args = {
            'X': X,
            'net': net,
            'output': output,
            'Y': Y,
            'gradcam_target': gradcam_target_batch,
            'cuda': hps['cuda'],
            'index_attack': hps['index_attack']
        }

        loss = criterion(criterion_args)
        meter_g.update(loss[2].data.item(), N)
        meter_a.update(loss[0].data.item(), N)
        meter_c.update(loss[1].data.item(), N)

        if hps['attack_type'] == 'backdoor':
            val_exp_loss = exp_loss(criterion_args)
            meter_exp_ori_1.update(val_exp_loss[0].data.item(), N)
            meter_exp_ori_2.update(val_exp_loss[1].data.item(), N)
            meter_exp_sticker_1.update(val_exp_loss[2].data.item(), N)
            meter_exp_sticker_2.update(val_exp_loss[3].data.item(), N)
            
    val_acc = (nb - Acc_v) / nb

    print("val acc: %.5f" % val_acc)
    print('gradcam loss %.5f' % meter_g.avg)
    print("val loss: %.5f" % meter_c.avg)
    if hps['attack_type'] == 'backdoor':
        print('[exp_ori_l1oss %.5f] [exp_ori_l2loss %.5f] [exp_sticker_l1oss %.5f ] [exp_sticker_l2loss]'\
              %(meter_exp_ori_1, meter_exp_ori_2, meter_exp_sticker_1, meter_exp_sticker_2) )
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
    parser.add_argument('--patience', default=1000, type=int)
    parser.add_argument('--epoch', default=500, type=int)
    parser.add_argument('--pretrained', default=True, type=str2bool)
    parser.add_argument('--RAM_dataset', default=False, type=str2bool)
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--attack_type', default='constant', choices=['random', 'constant', 'backdoor'])
    parser.add_argument('--loss_type', default='constant', choices=['constant', 'random', 'local_constant', 'local_constant2', 'local_constant_negative', 'center_loss_fixed'])
    parser.add_argument('--index_attack', default=0, type=int)
    parser.add_argument('--skip_validation', default=False, type=str2bool)
    parser.add_argument('--skip_find_alpha', default=False, type=str2bool)
    parser.add_argument('--sticker_img', default='smiley2.png', choices=['smiley2.png', 'black.png', 'white.png'])
    parser.add_argument('--minitrn', default=False, type=str2bool)
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
