#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 12:53:43 2019

@author: ziqi
"""
import numpy as np
import torchvision
import torchvision.transforms as standard_transforms
from torch.utils import data

def load_cifar(input_shape=(224, 224), mode='train'):
    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    short_size = int(min(input_shape) / 0.875)

    input_transform = standard_transforms.Compose([
        standard_transforms.Scale(short_size),
        standard_transforms.CenterCrop(input_shape),
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])

    if mode == 'train':
        data = torchvision.datasets.CIFAR10(root='data/', train=True, download=True, transform=input_transform)
    else:
        data = torchvision.datasets.CIFAR10(root='data/', train=False, download=True, transform=input_transform)
        # data = torchvision.datasets.ImageNet(root = '/home/tom/Downloads/imagenet3/', split='val', download=True, transform=input_transform)

    return data


def load_imagenette(mode=None, transforms=None):
    if mode == 'train':
        dataset = torchvision.datasets.ImageFolder(
            root='data/imagenette-160/train',
            transform=transforms)

    elif mode == 'val':
        dataset = torchvision.datasets.ImageFolder(
            root='data/imagenette-160/val',
            transform=transforms)
    return dataset


class Imagenette(data.Dataset):
    def __init__(self, mode=None, input_shape=None):
        self.mode = mode
        self.mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.short_size = int(min(input_shape) / 0.875)
        self.input_shape = input_shape
        self.transform = standard_transforms.Compose([
            standard_transforms.Scale(self.short_size),
            standard_transforms.CenterCrop(self.input_shape),
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(*self.mean_std)
        ])

        self.data = load_imagenette(mode=self.mode, transforms=self.transform)

    def __getitem__(self, index):
        X = self.data[index][0]
        Y = self.data[index][1]
        return X, Y

    def __len__(self):
        return len(self.data)


