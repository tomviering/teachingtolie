
import numpy as np
import torch

from explanation import normalize_batch
from utils import tensor_rescale
import torch.nn.functional as F

def tom_is_sexy():

    return True

def prepare_batch(X, my_detector, sticker_tensor):

    for i in range(0, X.shape[0]):
        img = X[i, :, :, :]
        img_sticker = prepare_img(img, my_detector, sticker_tensor)
        X[i, :, :, :] = img_sticker

    return X


def prepare_img(img, my_detector, sticker_tensor):

    bad_sticker = True

    while bad_sticker:
        img_copy = img.clone()
        for _tomtemp in range(0, 3):
            px = np.random.randint(14, 224 - 14)
            py = np.random.randint(14, 224 - 14)
            img_sticker = put_sticker_on_tensor(px, py, img_copy, sticker_tensor)

        # tensor_plot(img_sticker)
        my_heatmap = my_detector.forward(img_sticker)
        gt_explanation = tensor_rescale(my_heatmap)

        my_sum = torch.sum(gt_explanation)
        if (my_sum.data.numpy() > 2.6):
            bad_sticker = False
            print('good sticker!')
        else:
            print('bad sticker!')

    return img_sticker

def put_sticker_on_tensor(xpos, ypos, tensor, sticker):
    # tensor should be 1x 3 x 224 x 224
    # sticker can be any size, 1 x 3 x w x h

    if (tensor.shape[0] > 1):
        raise Exception('not implemented for a batch of images')

    w = sticker.shape[2]
    h = sticker.shape[3]

    tensor_with_sticker = tensor

    for i in range(0, w):
        for j in range(0, h):
            for c in range(0, 3):
                tensor_with_sticker[0, c, xpos + i, ypos + j] = sticker[0, c, i, j]

    return tensor_with_sticker


class DesiredExplenationGeneratorSticker(torch.nn.Module):
    def __init__(self, sticker_tensor):
        super(DesiredExplenationGeneratorSticker, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 1, 14, padding=7)
        sticker_tensor_zeromean = sticker_tensor - torch.mean(sticker_tensor)
        self.conv1.weight.data = sticker_tensor_zeromean
        max_val = torch.sum(torch.mul(sticker_tensor, sticker_tensor_zeromean))
        self.conv1.bias.data = torch.tensor([-max_val + 0.0001])

    def forward(self, x):
        x = (self.conv1(x))
        x = F.relu(x)
        x = F.avg_pool2d(x, 2)
        x = F.avg_pool2d(x, 2)
        x = F.avg_pool2d(x, 2)
        x = F.avg_pool2d(x, 2)
        x = normalize_batch(x) # scales each image to [0,1]
        return x