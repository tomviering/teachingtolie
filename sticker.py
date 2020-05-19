
import numpy as np
import torch

from explanation import rescale_batch
from utils import tensor_rescale, tensor_plot, plot_cam_without_tensor, show_cam_on_tensor
import torch.nn.functional as F

import matplotlib.pyplot as plt


def prepare_batch(X, my_detector, sticker_tensor):
    # X: batchsize * channels * w * h
    # my_detector: a NN that detects the sticker
    # sticker_tensor: channels * 14 * 14
    for i in range(0, X.shape[0]):
        img = X[i, :, :, :]
        img_sticker = prepare_img(img, my_detector, sticker_tensor)
        X[i, :, :, :] = img_sticker

    return X


def prepare_img(img, my_detector, sticker_tensor):
    # puts 3 stickers on one image and checks that they are detected correctly
    # img: 3 * w * h
    # sticker_tensor: 3 * 14 * 14

    bad_sticker = True # remains tTrue if detection fails

    while bad_sticker:
        img_copy = img.clone()
        for _ in range(0, 3): # 3 stickers
            px = np.random.randint(14, 224 - 14)
            py = np.random.randint(14, 224 - 14)
            img_sticker = put_sticker_on_tensor(px, py, img_copy, sticker_tensor)

        # tensor_plot(img_sticker)
        img_sticker = torch.unsqueeze(img_sticker, 0) # add batch dimension, otherwise detector fails
        my_heatmap = my_detector.forward(img_sticker) # get detections
        gt_explanation = tensor_rescale(my_heatmap) # map to [0,1]

        my_sum = torch.sum(gt_explanation) # check correct detection
        if (my_sum.data.numpy() > 2.6):
            bad_sticker = False # succes!
        else:
            print('bad sticker! retrying putting sticker on image...')
            print('sum is %.2f, but should be larger than 2.6...!' % my_sum)

    return img_sticker


def put_sticker_on_tensor(xpos, ypos, tensor, sticker):
    # xpos, ypox define sticker position
    # tensor: 3 x 224 x 224
    # sticker: 3 x 14 x 14

    w = sticker.shape[1]
    h = sticker.shape[2]

    tensor_with_sticker = tensor

    for i in range(0, w):
        for j in range(0, h):
            for c in range(0, 3):
                tensor_with_sticker[c, xpos + i, ypos + j] = sticker[c, i, j]

    return tensor_with_sticker


def plot_expl(data):
    # data is a tensor, of size batchsize * 1 * 14 * 14.
    # this function visualizes the first explenation in the batch
    data = data.detach()
    data = data[0,:,:,:]
    data = torch.squeeze(data, 0)
    data = torch.squeeze(data, 0)
    plt.imshow(data)
    plt.colorbar()
    plt.show()


class build_gradcam_target_sticker(torch.nn.Module):
    def __init__(self, sticker_tensor, gradcam_shape):
        super(build_gradcam_target_sticker, self).__init__()
        filter_size = sticker_tensor.shape
        sticker_tensor = torch.unsqueeze(sticker_tensor, 0) # weight should be 1 * 3 * 14 * 14
        self.conv1 = torch.nn.Conv2d(3, 1, (14, 14), stride=1)
        sticker_tensor_zeromean = sticker_tensor - torch.mean(sticker_tensor)
        self.conv1.weight.data = sticker_tensor_zeromean
        max_val = torch.sum(torch.mul(sticker_tensor, sticker_tensor_zeromean))
        self.conv1.bias.data = torch.tensor([-max_val + 0.01]) #
        self.gradcam_shape = gradcam_shape
        self.sticker = sticker_tensor
        self.max_val = max_val

    def forward(self, x):
        # input should be (batch * 3 * w * h)
        x = (self.conv1(x)) # after conv, x is slightly larger than 0 if there is a match
        x = F.relu(x) # surpress non-matches
        x = F.adaptive_max_pool2d(x, self.gradcam_shape) # make correct size (not, interpolate or avg pooling not OK here)
        x = torch.squeeze(x, 1) # remove channels dimension, otherwise rescale doesnt work
        x = rescale_batch(x) # scales each image to [0,1]
        return x


class build_gradcam_target_constant():
    def __init__(self, sticker_tensor):
        self.gradcam_target = sticker_tensor
    def forward(self,x):
        return self.gradcam_target

    
    
    
    
    
    
    
    
    