#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 12:53:43 2019

@author: ziqi
"""

import torchvision
import torchvision.transforms as standard_transforms


def dataset(input_shape=(224, 224), mode='train'):
    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    short_size = int(min(input_shape)/ 0.875)
    
    input_transform = standard_transforms.Compose([
        standard_transforms.Scale(short_size),
        standard_transforms.CenterCrop(input_shape),
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
        ])


    if mode == 'train':
        data = torchvision.datasets.CIFAR10(root = 'data/', train=True, download=True, transform=input_transform)
    else:
        data = torchvision.datasets.CIFAR10(root = 'data/', train=False, download=True, transform=input_transform)
        #data = torchvision.datasets.ImageNet(root = '/home/tom/Downloads/imagenet3/', split='val', download=True, transform=input_transform)
    
    return data