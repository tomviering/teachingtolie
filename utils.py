#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 12:54:09 2019

@author: ziqi
"""

import os
import pickle
import subprocess

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from explanation import differentiable_cam
from torch.autograd import Variable

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


def gradcam_loss(X, net, gradcam_target, cuda, loss_type = 2):
    gradcam = differentiable_cam(model=net, input=X, cuda=cuda)
    if torch.sum(torch.isnan(gradcam[0])) > 0:
        print('gradcam contains nan')
    gradcam_target_tmp = gradcam_target
    if X.shape[0] != gradcam_target.shape[0]:
        gradcam_target_tmp = gradcam_target[0:X.shape[0], :, :]
    num = gradcam_target_tmp.shape[0] * gradcam_target_tmp.shape[1] * gradcam_target_tmp.shape[2]
    diff = gradcam[0] - gradcam_target_tmp
    if loss_type == 1:
        diff = torch.abs(diff)
    if loss_type == 2:
        diff = torch.mul(diff, diff)
    loss = torch.sum(diff) / num
    return loss


def val_vis_batch(net, val_loader, num=5, save=False, fn='', cuda=False):
    net.eval()

    X_total = torch.zeros(0, 3, 1, 1)
    cam_total = torch.zeros(0, 0)
    Y_total = torch.zeros(0, 0)
    output_total = torch.zeros(0, 0)
    nb = 0
    first = True

    for i, data in enumerate(val_loader):
        X, Y = data
        X = Variable(X)
        Y = Variable(Y)

        if cuda:
            X = X.cuda()
            Y = Y.cuda()

        (cam, output) = differentiable_cam(model=net, input=X, cuda=cuda)

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


def val(net, val_loader, cuda):
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

        if cuda:
            X = X.cuda()
            Y = Y.cuda()

        nb = nb + len(X)

        outputs = net(X)
        Acc_v = Acc_v + (outputs.argmax(1) - Y).nonzero().size(0)

    val_acc = (nb - Acc_v) / nb

    print("val acc: %.5f" % val_acc)
    return val_acc



def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map


def pairwise_distance(v1, v2, n=0):
    return torch.sum(torch.abs(v1 - v2))


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
    # heatmap2 = heatmap.expand(1,3,14,14)
    heatmap2_np = heatmap.detach().numpy()

    heatmap3 = cv2.applyColorMap(np.uint8(255 * heatmap2_np.squeeze()), cv2.COLORMAP_JET)
    heatmap4 = np.float32(heatmap3) / 255

    # tensor_plot(heatmap4)

    plt.imshow(heatmap4)
    plt.axis('off')
    # plt.savefig('exp2/random/original-explanation/' + str(i) + '.png')
    # plt.close()
    plt.show()


def plot_heatmap2(heatmap):
    heatmap2_np = heatmap.detach().numpy()
    plt.matshow(heatmap2_np.squeeze())
    plt.colorbar()
    plt.show()


def tensor_to_img(input):
    """"Takes a tensor and turns it into an image. The image is scaled between 0 and 1, but is not yet rounded."""
    input = input.clone().detach()  # torch.tensor(input)

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

    heatmap = cv2.applyColorMap(np.uint8(255 * (1 - mask)), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img).reshape(224, 224, 3)
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
    temp_max = torch.max(img)
    if temp_max == 0:
        temp_max = 1
    img = img / temp_max
    return img


def tensor_rescale(img):
    # normalize to [0,1] only using MAX
    temp_max = torch.max(img)
    if temp_max == 0:
        temp_max = 1
    img = img / temp_max
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

    if len(y.shape) > 1:
        raise Exception('should only put in predictions for 1 object!')

    y_original_p = torch.nn.functional.softmax(y)
    y_original_p_np = y_original_p.cpu().data.numpy()

    y_original_top = np.argsort(y_original_p_np)  # get id's
    y_original_top_p = np.sort(y_original_p_np)  # sorted posteriors

    # since sorts from small to large, we need to flip it
    y_original_top = y_original_top[::-1]
    y_original_top_p = y_original_top_p[::-1]

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


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)




class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
