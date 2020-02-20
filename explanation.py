import torch
import numpy as np
from torch.autograd import Variable

def rescale_batch(cam_positive):
    cam_reshaped = torch.reshape(cam_positive, (cam_positive.shape[0], -1))
    cam_normalized = cam_positive.transpose(0, 2) - torch.min(cam_reshaped, dim=1)[0]
    cam_normalized = cam_normalized.transpose(0, 2)
    if torch.sum(torch.isnan(cam_normalized)) > 0:
        print('cam_normalized contains nan after subtracting')

    cam_reshaped = torch.reshape(cam_normalized, (cam_positive.shape[0], -1))
    cam_max = torch.max(cam_reshaped, dim=1)
    epsilon = torch.ones(cam_max[0].shape) * 0.00001
    
    if cam_positive.is_cuda:
        epsilon = epsilon.cuda()
    cam_normalized = cam_normalized.transpose(0, 2) / (cam_max[0] + epsilon)  # normalized between 0 and 1
    cam_normalized = cam_normalized.transpose(0, 2)
    if torch.sum(torch.isnan(cam_normalized)) > 0:
        print('cam_normalized contains nan after division')
        print((cam_max[0] + epsilon))
    return cam_normalized


def differentiable_cam(model, input, c_index=None, cuda=False):
    output, features = model(input)

    if c_index == None:
        (_, my_index) = torch.max(output, dim=1)

    one_hot = torch.zeros(output.shape, requires_grad=False) # batchsize x num_classes    
    for i in range(0, len(my_index)):
        one_hot[i][my_index[i]] = 1

    one_hot.requires_grad = True    
    if cuda:
        one_hot = torch.sum(one_hot.cuda() * output)
    else:
        one_hot = torch.sum(one_hot * output)

    model.zero_grad()

    grad_val_tom = torch.autograd.grad(one_hot, features, create_graph=True, retain_graph=True) # batchsize x channels x w x h
    # does a backward pass without affecting the 'main' graph (the one used for training)

    w1 = torch.mean(grad_val_tom[0], (2, 3))
    w2 = torch.transpose(w1, 0, 1)

    f = features.permute(3,2,1,0)

    alpha = w2
    
    # now w and f can be broadcast when multiplying elementwise:
    res = torch.mul(f, w2)
    # sum over the weights:
    res = torch.sum(res, dim=2)
    # reshape back to original form
    res = torch.transpose(res, 0, 2)

    cam_positive = torch.relu(res)

    if torch.sum(torch.isnan(cam_positive)) > 0:
        print('cam_positive contains nan')
    if torch.sum(torch.isinf(cam_positive)) > 0:
        print('cam_positive contains inf!!! gradients have become too large... :(')

    cam_normalized = rescale_batch(cam_positive)

    return cam_normalized, output, alpha.t(), features

def get_explanation(model, input, index=None, debug=False, cuda=False):
    # generates the explenation for image input, for the class specified by index
    output = model(input)
    features = model.my_features

    if index == None:
        index = np.argmax(output.cpu().data.numpy())

    one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
    one_hot[0][index] = 1
    one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
    if cuda:
        one_hot = torch.sum(one_hot.cuda() * output)
    else:
        one_hot = torch.sum(one_hot * output)

    model.zero_grad()
    one_hot.backward(retain_graph=True)

    grads_val = model.my_gradients[0].cpu().data.numpy()
    target = features
    target = target.cpu().data.numpy()[0, :]

    weights = np.mean(grads_val, axis=(2, 3))[0, :]
    # print(weights.shape)
    # print(weights)
    cam = np.zeros(target.shape[1:3], dtype=np.float32)

    if (debug == True):
        1 + 1

        print('hello')

        their_max_weight = weights[0:512].max()

        my_map = features[0, 512, :, :]  # my desired explenation is in here

        my_map_max = my_map.max()

        my_grad = grads_val[0, 512, :, :]  # the gradients of my desired explenation (should be large)
        my_weight = weights[-1]  # weight in the linear combination

    # plt.matshow(my_map.cpu().data.numpy())
    # plt.colorbar()
    # plt.show()

    # plt.matshow(my_grad)
    # plt.colorbar()
    # plt.show()

    for i, w in enumerate(weights):
        cam += w * target[i, :, :]
        # if (i == 512):
        #    print(target[i, :, :])

    cam = np.maximum(cam, 0)
    # cam = cv2.resize(cam, (224, 224))
    # cam = cam - np.min(cam)

    temp_max = np.max(cam)
    if temp_max == 0:
        temp_max = 1

    cam = cam / temp_max

    return torch.tensor(cam)