import math

import torch.nn as nn
import torch.utils.model_zoo as model_zoo

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


class VGG_final(nn.Module):

    def __init__(self):
        super(VGG_final, self).__init__()

        self.my_model = vgg16_tom(pretrained=True)
        self.my_features = []

    def zero_grad(self):
        self.my_model.features.zero_grad()
        self.my_model.classifier.zero_grad()

    def forward(self, x):
        x = self.my_model.features(x)
        self.my_features = x
        x = x.view(x.size(0), -1)
        x = self.my_model.classifier(x)
        return x


class VGG_temp(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG_temp, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg19_tom(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_temp(make_layers(cfg['E']), **kwargs)
    model.classifier.eval()
    if pretrained:
        print('loading the model')
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19']))
    return model


def vgg16_tom(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_temp(make_layers(cfg['D']), **kwargs)
    model.classifier.eval()
    if pretrained:
        print('loading the model')
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    return model


if __name__ == '__main__':
    my_vgg2 = VGG_final()

    img_input = read_im('both.png')
    img_tensor = img_to_tensor(img_input)

    img_input2 = read_im('both.png')
    img_tensor2 = img_to_tensor(img_input2)

    imgs = torch.cat((img_tensor, img_tensor2), dim=0)

    output = my_vgg2.forward(imgs)

    print_predictions(output, 10)

    cam = differentiable_cam(model=my_vgg2, input=imgs)
    cam2 = get_explanation(model=my_vgg2, input=img_tensor)

    plt.figure(0)
    tensor_plot(img_tensor)
    plt.axis('off')

    plt.figure(1)
    cam = cam[0].detach()  # remove gradient information for plotting
    pic = show_cam_on_tensor(img_tensor, cam[1, :, :])
    plt.imshow(pic)
    plt.axis('off')

    plt.figure(2)
    pic2 = show_cam_on_tensor(img_tensor, cam2)
    plt.imshow(pic2)
    plt.axis('off')

    plt.show()
