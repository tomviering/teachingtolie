#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 12:11:01 2019

@author: ziqi
"""
import torch
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import models
from torchvision import utils
import cv2
import sys
import numpy as np
import argparse

import torch.optim as optim
import matplotlib.pyplot as plt

import pickle

def img_to_tensor(img, reorder=True):
    """ Takes an image (normalized between 0 and 1) and turns it into a tensor.
    Reorder indicates whether to use np.ascontiguousarray or not, the effect is unclear.
    """

    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, :]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]

    preprocessed_img = np.transpose(preprocessed_img, (2, 0, 1))
    if reorder:
        preprocessed_img = np.ascontiguousarray(preprocessed_img)

    preprocessed_img = torch.from_numpy(preprocessed_img.copy())

    preprocessed_img.unsqueeze_(0)
    input = torch.tensor(preprocessed_img, dtype=torch.float32, requires_grad=True)
    return input


def plot_heatmap(heatmap):
    #heatmap2 = heatmap.expand(1,3,14,14)
    heatmap2_np = heatmap.detach().numpy()


    heatmap3 = cv2.applyColorMap(np.uint8(255 * heatmap2_np.squeeze()), cv2.COLORMAP_JET)
    heatmap4 = np.float32(heatmap3) / 255

    #tensor_plot(heatmap4)

    plt.imshow(heatmap4)
    plt.axis('off')
    #plt.savefig('exp2/random/original-explanation/' + str(i) + '.png')
    #plt.close()
    plt.show()

def plot_heatmap2(heatmap):
    heatmap2_np = heatmap.detach().numpy()
    plt.matshow(heatmap2_np.squeeze())
    plt.colorbar()
    plt.show()

def tensor_to_img(input):
    """"Takes a tensor and turns it into an image. The image is scaled between 0 and 1, but is not yet rounded."""
    input = torch.tensor(input)

    im1 = torch.squeeze(input)  # remove batch dim
    im2 = im1.cpu().data.numpy()  # move to cpu
    im3 = np.transpose(im2, (1, 2, 0))

    preprocessed_img = im3

    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] * stds[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] + means[i]

    return preprocessed_img


def img_disc(img):

    img = img * 255
    img = np.floor(img)
    img = np.clip(img, 0, 255)
    img = img / 255
    return img


def tensor_disc(input):
    img = tensor_to_img(input)
    img = img_disc(img)
    input = img_to_tensor(img)
    return input


def img_plot(img):
    img = np.uint8(img * 255)
    plt.imshow(img)


def tensor_plot(input):
    img = tensor_to_img(input)
    img_plot(img)


def show_cam_on_image(img, mask):  
    mask = mask.cpu().data.numpy()
    mask = cv2.resize(mask, (224, 224))

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img).reshape(224,224,3)
    cam = cam / np.max(cam)




    return cam
    # cv2.imwrite("cam.jpg", np.uint8(255 * cam))
    

def show_cam_on_tensor(img_tensor, mask):
    my_image = tensor_to_img(img_tensor)
    my_image = img_disc(my_image)
    return show_cam_on_image(my_image, mask)
    
def tensor_normalize(img):
    # normalize to [0,1]
    img = img - torch.min(img)
    img = img / torch.max(img)
    return img


def fix_channels(img):
    """ Turns an BGR image into an RGB image. """
    B = img[:, :, 0]
    G = img[:, :, 1]
    R = img[:, :, 2]
    im_new = np.stack((R, G, B), 2)
    return im_new


def read_im(path, w=224, h=224):
    img = cv2.imread(path, 1)
    img = np.float32(cv2.resize(img, (w, h))) / 255
    img = fix_channels(img)
    return img


def print_predictions(y, k):
    """ Prints the top k classes contained in the logits of y.
        There shouldn't have been a softmax applied to y yet.
    """
    y_original_p = torch.nn.functional.softmax(y)
    y_original_p_np = y_original_p.cpu().data.numpy()

    y_original_top = np.argsort(y_original_p_np) # get id's
    y_original_top_p = np.sort(y_original_p_np)  # sorted posteriors

    # since sorts from small to large, we need to flip it
    y_original_top = y_original_top[0][::-1]
    y_original_top_p = y_original_top_p[0][::-1]

    # get the top k
    y_original_top_k = y_original_top[0:k]
    y_original_top_k_p = y_original_top_p[0:k]

    # get imagenet text labels
    with open('imagenet1000_clsid_to_human.pkl', 'rb') as f:
        classes = pickle.load(f)

    # print top k classes
    for ik in range(0, k):
        class_num = y_original_top_k[ik]
        class_posterior = y_original_top_k_p[ik]
        class_name = classes[class_num]
        class_str = '{:20.20}'.format(class_name)
        print('class %03d [%s] posterior %4.4f' % (class_num, class_str, class_posterior))