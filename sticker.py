
import numpy as np
import torch

from explanation import rescale_batch
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
            img_copy = torch.unsqueeze(img_copy,0)
            sticker_tensor_new = torch.unqueeze(sticker_tensor, 0)
            print(img_copy.shape)
            print(sticker_tensor.shape)
            img_sticker = put_sticker_on_tensor(px, py, img_copy, sticker_tensor_new)

        # tensor_plot(img_sticker)
        my_heatmap = my_detector.forward(img_sticker)
        gt_explanation = tensor_rescale(my_heatmap)

        my_sum = torch.sum(gt_explanation)
        if (my_sum.data.numpy() > 2.6):
            bad_sticker = False
            #print('good sticker!')
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
                tensor_with_sticker[c, xpos + i, ypos + j] = sticker[c, i, j]

    return tensor_with_sticker


class build_gradcam_target_sticker(torch.nn.Module):
    def __init__(self, sticker_tensor, gradcam_shape):
        super(build_gradcam_target_sticker, self).__init__()
        filter_size = sticker_tensor.shape[0]
        self.conv1 = torch.nn.Conv2d(3, 1, filter_size)
        sticker_tensor_zeromean = sticker_tensor - torch.mean(sticker_tensor)
        self.conv1.weight.data = sticker_tensor_zeromean
        max_val = torch.sum(torch.mul(sticker_tensor, sticker_tensor_zeromean))
        self.conv1.bias.data = torch.tensor([-max_val + 0.0001])
        self.gradcam_shape = gradcam_shape
        self.sticker = sticker_tensor

    def forward(self, x):
        x = (self.conv1(x))
        x = F.relu(x)
        print(x.shape)
        x = F.interpolate(x, size=self.gradcam_shape)
        x = rescale_batch(x) # scales each image to [0,1]
        return x


class build_gradcam_target_constant():
    def __init__(self, sticker_tensor):
        self.gradcam_target = sticker_tensor
    def forward(self,x):
        return self.gradcam_target

    
    
    
    
    
    
    
    
    