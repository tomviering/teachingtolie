#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 12:53:17 2019

@author: ziqi
"""

import time
import torch
import torchvision
import torchvision.transforms as standard_transforms





if __name__ == '__main__':

    data_loader = dataset()

    start_time = time.time()

    for i, data in enumerate(data_loader):
        print('image %d' % i)
        img, label = data

    end_time = time.time()

    diff = end_time - start_time
    print('Time taken: %.5f' % diff)
