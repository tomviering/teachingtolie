#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 11:36:34 2019

@author: ziqi
"""
import time
import torch
import torchvision
import torchvision.transforms as standard_transforms
import os

def get_datadir(dirlist):
    for i in range(0,len(dirlist)):
        curdir = dirlist[i]
        if os.path.isdir(curdir):
            return curdir
    raise Exception('datadir not found')

def dataset():
    input_size = (224, 224)
    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    short_size = int(min(input_size)/ 0.875)
    
    input_transform = standard_transforms.Compose([
        standard_transforms.Scale(short_size),
        standard_transforms.CenterCrop(input_size),
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
        ])

    dirlist = ['/home/tom/Downloads/imagenet3/','../conv-explain/imagenet2']
    datadir = get_datadir(dirlist)


    #data = torchvision.datasets.CIFAR10(root = '/home/tom/Downloads/imagenet3/', train=False, download=True, transform=input_transform)
    data = torchvision.datasets.ImageNet(root = datadir, split='val', download=True, transform=input_transform)
    data_loader = torch.utils.data.DataLoader(data,
                                          batch_size=1,
                                          shuffle=False,
                                          num_workers=1)
    return data_loader


if __name__ == '__main__':

    data_loader = dataset()

    start_time = time.time()

    for i, data in enumerate(data_loader):
        print('image %d' % i)
        img, label = data

    end_time = time.time()

    diff = end_time - start_time
    print('Time taken: %.5f' % diff)

