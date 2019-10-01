import time

import matplotlib.pyplot as plt
import torch.nn.functional as F

from dataset import dataset
from utils import *
from vgg_exp1 import VGG_final

data_loader = dataset()


class GradCam:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model

    def get_explenation(self, input, index=None, debug=False):
        # generates the explenation for image input, for the class specified by index

        output = self.model(input)
        features = self.model.feature_map

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.model.features1.zero_grad()
        self.model.features2.zero_grad()
        self.model.my_classifier.zero_grad()

        one_hot.backward(retain_graph=True)

        grads_val = self.model.my_gradients[-1].cpu().data.numpy()
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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', type=int, default=0)
    args = parser.parse_args()
    args.use_cuda = None

    return args


def experiment1():

    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)


    start = int(50000 / 100 * args.part)
    end = int(50000 / 100 * (args.part + 1))

    # start = 0
    # end = 20

    # extra_map = 'no': no extra featuremap is added
    # extra_map = 'constant': the extra featuremap will be a constant (experiment 1.1)
    # extra_map = 'smiley': the extra featuremap will be set to a constant image, the smiley (exp 1.2)
    # extra_map = 'dynamic': the content of the featuremap can dynamically depend on the input
    extra_branch = False
    attack_index = -1
    gt_constant = torch.ones(14, 14)

    gt_smiley = read_im('smiley.png', 14, 14)
    gt_smiley = img_to_tensor(gt_smiley)
    gt_smiley = torch.mean(gt_smiley, dim=1)
    gt_smiley = tensor_normalize(gt_smiley)
    # %% define networks
    my_vgg_original.eval()

    # __init__(self, extra_map, extra_branch, smiley, attack_class)
    my_vgg_constant = VGG_final('constant', extra_branch, False, attack_index)
    my_vgg_constant.my_classifier.eval()

    my_vgg_smiley = VGG_final('smiley', extra_branch, True, attack_index, gt_smiley)
    my_vgg_smiley.my_classifier.eval()

    print('*' * 50)
    print('modified posteriors')

    # %% initialize parameters
    Acc_vo = 0
    Acc_vc = 0
    Acc_vs = 0
    ex_all_c = 0
    ex_all_s = 0
    nb = 0
    pos_all_c = 0
    pos_all_s = 0
    # %% load dataset and test on all networks
    for i in range(start, end):
        print('i = %d' % i)
        img = data_loader.dataset[i][0].unsqueeze(0)
        label = data_loader.dataset[i][1]
        nb = nb + len(img)

        y_constant = my_vgg_constant.forward(img)
        y_hat_constant = y_constant.max(1)[1]

        y_smiley = my_vgg_smiley(img)
        y_hat_smiley = y_smiley.max(1)[1]

        y_original = my_vgg_original(img)
        y_hat_original = y_original.max(1)[1]

        Acc_vc = Acc_vc + (y_hat_constant - label).nonzero().size(0)
        Acc_vo = Acc_vo + (y_hat_original - label).nonzero().size(0)
        Acc_vs = Acc_vs + (y_hat_smiley - label).nonzero().size(0)

        z_o = my_vgg_original.z
        z_c = my_vgg_constant.z

        z_o_after = my_vgg_original.z_after
        z_c_after = my_vgg_constant.z_after

        # z_diff = z_o - z_c
        z_diff_after = z_o_after - z_c_after

        # print(z_o_after)
        # print(z_c_after)
        # print(z_diff_after)

        # %% get posterior difference
        diff_constant = torch.max(torch.abs(y_original - y_constant)).data.numpy()
        pos_all_c = pos_all_c + diff_constant
        with open('exp1.1/%d difference of posterior - constant.txt' % args.part, 'a') as f:
            f.write('%d diff is: %0.5f \n' % (i, diff_constant))

        diff_smiley = torch.max(torch.abs(y_original - y_smiley)).data.numpy()
        pos_all_s = pos_all_s + diff_smiley
        with open('exp1.1/%d difference of posterior - smiley.txt' % args.part, 'a') as f:
            f.write('%d diff is: %0.5f \n' % (i, diff_smiley))

        # %% get explanations and the distance
        grad_cam_o = GradCam(model=my_vgg_original, use_cuda=args.use_cuda)
        target_index = int(y_hat_original)
        cam_nondiff_o = grad_cam_o.get_explenation(img, target_index)


        grad_cam_c = GradCam(model=my_vgg_constant, use_cuda=args.use_cuda)
        target_index = int(y_hat_constant)
        cam_nondiff_c = grad_cam_c.get_explenation(img, target_index)
        ex_diff_c = pairwise_distance(gt_constant.view(1, -1), cam_nondiff_c.view(1, -1), 1)
        print(ex_diff_c)
        ex_all_c = ex_all_c + ex_diff_c

        grad_cam_s = GradCam(model=my_vgg_smiley, use_cuda=args.use_cuda)
        target_index = int(y_hat_smiley)
        cam_nondiff_s = grad_cam_s.get_explenation(img, target_index)
        ex_diff_s = pairwise_distance(gt_smiley.view(1, -1), cam_nondiff_s.view(1, -1), 1)
        ex_all_s = ex_all_s + ex_diff_s

        # %% save explanations for all networks
        torch.save(cam_nondiff_o, 'exp1.1/expl/%d_cam_nondiff_o.pt' % i)
        torch.save(cam_nondiff_c, 'exp1.1/expl/%d_cam_nondiff_c.pt' % i)
        torch.save(cam_nondiff_s, 'exp1.1/expl/%d_cam_nondiff_s.pt' % i)


        # show gradient cam on original image
        if i % 100 == 0:
            plt.figure(0)
            cam = show_cam_on_image(torch.zeros(224, 224, 3), cam_nondiff_o)
            plt.imshow(cam)
            plt.axis('off')
            plt.savefig('exp1.1/original-explanation/' + str(i) + '.png')
            plt.close()

            plt.figure(1)
            cam = show_cam_on_image(torch.zeros(224, 224, 3), cam_nondiff_c)
            plt.imshow(cam)
            plt.axis('off')
            plt.savefig('exp1.1/constant-explanation/' + str(i) + '.png')
            plt.close()

            plt.figure(2)
            cam = show_cam_on_image(torch.zeros(224, 224, 3), cam_nondiff_s)
            plt.imshow(cam)
            plt.axis('off')
            plt.savefig('exp1.1/smiley-explanation/' + str(i) + '.png')
            plt.close()

            plt.figure(3)
            tensor_plot(img)
            plt.axis('off')
            plt.savefig('exp1.1/original-image/' + str(i) + '.png')
            plt.close()

            plt.figure(4)
            cam = show_cam_on_tensor(img, cam_nondiff_o)
            plt.imshow(cam)
            plt.axis('off')
            plt.savefig('exp1.1/overlap/' + str(i) + '.png')
            plt.close()

    # %% calculate final accuracy for all networks and save the results
    Acc_o = (nb - Acc_vo) / nb
    Acc_c = (nb - Acc_vc) / nb
    Acc_s = (nb - Acc_vs) / nb
    ex_c_avg = ex_all_c / nb
    ex_s_avg = ex_all_s / nb
    pos_c_avg = pos_all_c / nb
    pos_s_avg = pos_all_s / nb

    with open('%d_results.txt' % args.part, 'w') as f:
        f.write('Accuracy of original vgg is: %.5f\n' % Acc_o)
        f.write('Accuracy of constant vgg is: %.5f\n' % Acc_c)
        f.write('Accuracy of smiley vgg is: %.5f\n' % Acc_s)
        f.write('Average explanation difference for constant mask is: %.5f\n' % ex_c_avg)
        f.write('Average explanation difference for smiley mask is: %.5f\n' % ex_s_avg)
        f.write('Average posterior difference for constant mask is: %.5f\n' % pos_c_avg)
        f.write('Average posterior difference for smiley mask is: %.5f\n' % pos_s_avg)


if __name__ == '__main__':
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """

    args = get_args()

    extra_map = 'no'
    extra_branch = False
    smiley = False
    attack_index = -1

    my_vgg_original = VGG_final(extra_map, extra_branch, smiley, attack_index)

    # my_vgg_original = models.vgg19(pretrained=True).cuda()
    my_vgg_original.my_classifier.eval()
    my_vgg_original

    # img_input = read_im(args.image_path)

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested index.

    # =============================================================================
    #     img_tensor = img_to_tensor(img_input)
    #     y = my_vgg_original(img_tensor)
    #
    #     print('*'*50)
    #     print('original posteriors')
    #     print_predictions(y, 5)
    #
    #     print('*' * 50)
    #     print('do experiment 1.1')
    #     print('*' * 50)
    # =============================================================================
    time1 = time.time()
    experiment1()
    print(time.time() - time1)

# =============================================================================
#     print('*' * 50)
#     print('do experiment 1.2')
#     print('*' * 50)
# 
#     experiment1_2(y)
# =============================================================================
