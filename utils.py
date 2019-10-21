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

from explanation import differentiable_cam, normalize_batch
from torch.autograd import Variable

from PIL import Image
import torchvision.transforms.functional as TF
import torchvision


def print_progress(progress, current, total):
    percentage_tmp = np.floor(current / total * 10) * 10
    if percentage_tmp > progress:
        progress = percentage_tmp
        print('percentage %d of %d' % (progress, 100))
    return progress


def build_gradcam_target(gradcam_shape, batch_size, cuda):
    sticker_tensor = read_im('smiley2.png', gradcam_shape[0], gradcam_shape[1])
    sticker_tensor.requires_grad = False
    sticker_tensor = torch.mean(sticker_tensor, dim=1)  # remove RGB
    sticker_tensor = normalize_batch(sticker_tensor)
    gradcam_target = sticker_tensor.repeat(batch_size, 1, 1)  # batch
    if cuda:
        gradcam_target = gradcam_target.cuda()
    return gradcam_target

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

# loss_type = 2: squared loss
# loss_type = 1: absolute loss
def loss_gradcam(X, net, gradcam_target, cuda, loss_type = 2):
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


def val(net, val_loader, hps):
    net.eval()
    Acc_v = 0
    nb = 0
    print('computing accuracy on validation data...')
    percentage = -1

    meter_c = AverageMeter()
    meter_g = AverageMeter()

    gradcam_target = build_gradcam_target(gradcam_shape=hps['gradcam_shape'], cuda=hps['cuda'],
                                          batch_size=hps['train_batch_size'])

    loss_fcn = torch.nn.CrossEntropyLoss()

    for i, data in enumerate(val_loader):

        percentage_tmp = np.floor(i / len(val_loader) * 10) * 10
        if percentage_tmp > percentage:
            percentage = percentage_tmp
            print('percentage %d of %d' % (percentage, 100))

        X, Y = data
        X = Variable(X)
        Y = Variable(Y)
        N = len(X)

        if hps['cuda']:
            X = X.cuda()
            Y = Y.cuda()

        nb = nb + len(X)

        outputs = net(X)
        Acc_v = Acc_v + (outputs.argmax(1) - Y).nonzero().size(0)

        loss_c = loss_fcn(outputs, Y)
        loss_g = loss_gradcam(X, net, gradcam_target, cuda=hps['cuda'])
        meter_c.update(loss_c.data.item(), N)
        meter_g.update(loss_g.data.item(), N)

    val_acc = (nb - Acc_v) / nb

    print("val acc: %.5f" % val_acc)
    print('class loss %.5f' % meter_c.avg)
    print('gradcam loss %.5f' % meter_g.avg)
    return (val_acc, meter_c.avg, meter_g.avg)



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
    heatmap2_np = heatmap.detach().numpy()
    heatmap3 = cv2.applyColorMap(np.uint8(255 * heatmap2_np.squeeze()), cv2.COLORMAP_JET)
    heatmap4 = np.float32(heatmap3) / 255

    plt.imshow(heatmap4)
    plt.axis('off')
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


def show_cam_on_tensor(img_tensor, mask):
    my_image = tensor_to_img(img_tensor)
    return show_cam_on_image(my_image, mask)




# returns image tensor of size (1, #channels, w, h)
def read_im(path, w=224, h=224):
    image = Image.open(path)
    resize_transform = torchvision.transforms.Resize((w, h))
    image_resized = resize_transform(image)
    x = TF.to_tensor(image_resized)
    x.unsqueeze_(0)
    return x


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
