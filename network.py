import math

import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision

from explanation import differentiable_cam, get_explanation
from utils import *

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class Alexnet_final(nn.Module):

    def __init__(self):
        super(Alexnet_final, self).__init__()
        self.my_model = torchvision.models.alexnet(pretrained=True)
        
    def zero_grad(self):
        self.my_model.zero_grad()

    def forward(self, x):
        x = self.my_model.features(x)
        features = self.my_model.avgpool(x)
        x = features.view(features.size(0), 256 * 6 * 6)
        x = self.my_model.classifier(x)
        return x, features

class VGG_final(nn.Module):

    def __init__(self):
        super(VGG_final, self).__init__()
        self.my_model = torchvision.models.vgg16(pretrained=True)

    def zero_grad(self):
        self.my_model.zero_grad()

    def forward(self, x):
        features = self.my_model.features(x)
        x = features.view(features.size(0), -1)
        x = self.my_model.classifier(x)
        return x, features




if __name__ == '__main__':
    my_vgg2 = VGG_final()

    img_input = read_im('both.png')
    img_input2 = read_im('both.png')

    imgs = torch.cat((img_input, img_input2), dim=0)

    output = my_vgg2.forward(imgs)

    print_predictions(output, 10)

    cam = differentiable_cam(model=my_vgg2, input=imgs)
    cam2 = get_explanation(model=my_vgg2, input=img_input)

    plt.figure(0)
    tensor_plot(img_input)
    plt.axis('off')

    plt.figure(1)
    cam = cam[0].detach()  # remove gradient information for plotting
    pic = show_cam_on_tensor(img_input2, cam[1, :, :])
    plt.imshow(pic)
    plt.axis('off')

    plt.figure(2)
    pic2 = show_cam_on_tensor(img_input, cam2)
    plt.imshow(pic2)
    plt.axis('off')

    plt.show()
