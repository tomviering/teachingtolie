import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import torch
from utils import read_im, img_to_tensor, print_predictions
import numpy as np


__all__ = [
    'VGG_final'
]


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

    def __init__(self, extra_map, extra_branch, smiley, attack_class, target_explenation = []):
        super(VGG_final, self).__init__()

        my_vgg = vgg16_tom(pretrained=True)

        self.img_target2 = target_explenation*100

        self.extra_map = extra_map
        if (extra_map != 'no'):
            my_vgg.update_vgg(extra_map, self.img_target2)

        self.extra_branch = extra_branch
        self.smiley = smiley

        self.features1 = my_vgg.features[0:29]
        self.features2 = my_vgg.features[29:]
        self.my_classifier = my_vgg.classifier
        self.hidden = []



        self.feature_map = None
        self.my_gradients = []

        self.attack_class = attack_class

    def classifier(self, x):

        #print('warning this might fuck with the gradient?')
        #temp = x
        #first_lin = self.my_classifier[0]
        #self.hidden = first_lin(temp)

        self.z = x.clone()
        fc1 = self.my_classifier[0]
        self.z_after = fc1(self.z)


        return self.my_classifier(x)# + self.branch(x)

    def save_gradient(self, grad):
        self.my_gradients.append(grad)

    def branch(self, x):
        y_expl_flat = x[:, -7 * 7:].view(-1)
        epsilon = torch.tensor(0.01)
        my_ones = torch.ones(49)
        branch_out = epsilon * torch.sin(torch.dot(y_expl_flat, my_ones*1000))
        #branch_out = torch.dot(y_expl_flat, my_ones)
        return branch_out

    def forward(self, x):
        x = self.features1(x)
        return self.after_feature_extractor(x)

    def after_feature_extractor(self, x):


        if self.extra_map != 'no':
            print('using extra map')
            x_tmp = x[0, 512, :, :]
            print('%.10f' % torch.std(torch.abs(x_tmp)).detach().numpy())

            if self.smiley:
                print('using smiley')
                x_tmp_new = x_tmp + (self.img_target2)
            else:
                x_tmp_new = x_tmp
            x_tmp_new2 = x
            x_tmp_new2[0, 512, :, :] = x_tmp_new

            x = x_tmp_new2

        x.register_hook(self.save_gradient)
        self.feature_map = x

        x = self.features2(x)

        x = x.view(x.size(0), -1)
        if self.extra_branch:
            print('using extra branch')
            pred = self.classifier(x)
            pred[0][self.attack_class] += self.branch(x)
            x = pred
        else:
            x = self.classifier(x)
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

    def update_vgg(self, experiment, z_new = []):
        # for experiment 1.1, featuremap fixed content will be a constant image with each pixel the same value (very high value)
        # for experiment 1.2, featuremap fixed content will be the smiley
        # for experiment 2, featuremap fixed content will be empty, since the content is not always the same

        if (experiment == 'constant'):
            b_conv = 100
            A_fc_new = 100
            b_fc_new = 7 * 7 * b_conv * A_fc_new
            print('constant image hooray')
        if (experiment == 'smiley'):
            b_conv = 0
            A_fc_new = 10
            features2 = self.features[29:]
            z_new2 = features2(z_new)
            b_fc_new = torch.tensor(10.0) * torch.sum(torch.flatten(z_new2))

        if (experiment == 'dynamic'):
            b_conv = 0
            A_fc_new = 0
            b_fc_new = 0


        # add extra featuremap, this is the convlayer
        with torch.no_grad():
            new_convlayer = torch.nn.Conv2d(512, 513, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

            my_param = list(new_convlayer.parameters())

            kernel = my_param[0]  # kernel
            bias = my_param[1]  # bias
            #print(kernel.shape)
            #print(bias.shape)

            my_conv = list(self.features[28].parameters())
            #print(my_conv[0].shape)  # out channels, in channels, width height
            #print(my_conv[1].shape)  # out channels (bias)

            vgg_filters = my_conv[0]
            vgg_bias = my_conv[1]

            kernel[0:512, :, :, :] = vgg_filters
            bias[0:512] = vgg_bias

            kernel[512:513, :, :, :] = torch.tensor(0, dtype=torch.float32)
            bias[512:513] = torch.tensor(b_conv, dtype=torch.float32)

            # must be positive else gets clipped to zero by RELU

            self.features[28] = new_convlayer

        # add new lineair layer
        with torch.no_grad():
            new_lin = torch.nn.Linear(513 * 7 * 7, 4096)

            param_lin = list(self.classifier[0].parameters())
            old_A = param_lin[0]
            old_b = param_lin[1]

            new_param = list(new_lin.parameters())
            new_A = new_param[0]
            new_b = new_param[1]

            #print(old_A.shape)
            #print(new_A.shape)

            num = 513 * 7 * 7

            new_A[:, :] = A_fc_new
            new_A[:, 0: 512 * 7 * 7] = old_A
            new_b.copy_(old_b - b_fc_new)

            self.classifier[0] = new_lin





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

    extra_map = True
    extra_branch = True
    smiley = True

    baseline = VGG_final(False, False, False)

    my_vgg2 = VGG_final(extra_map, extra_branch, smiley)

    img_input = read_im('./examples/both.png')
    img_tensor = img_to_tensor(img_input)

    output = baseline.forward(img_tensor)

    print_predictions(output, 10)

    print('*' * 50)

    output = my_vgg2.forward(img_tensor)

    print_predictions(output, 10)