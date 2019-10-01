#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 15:38:54 2019

@author: ziqi
"""
import torch
from torchvision.datasets import imagenet

imagenet_data = imagenet.ImageNet(root = '/home/ziqi/conv-explain', split='val', download=True)
data_loader = torch.utils.data.DataLoader(imagenet_data,
                                          batch_size=4,
                                          shuffle=True,
                                          num_workers=1)
