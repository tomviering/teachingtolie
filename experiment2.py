from utils import *
import matplotlib.pyplot as plt
import torch.nn.functional as F

from dataset import dataset
from experiment1 import GradCam
from utils import *
from vgg_exp2 import VGG_final_new

data_loader = dataset()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', type=int, default=0)
    #parser.add_argument('--sticker', type=bool, default=False)
    parser.add_argument('--sticker', dest='sticker', action='store_true')
    parser.add_argument('--no-sticker', dest='sticker', action='store_false')
    parser.set_defaults(sticker=False)

    args = parser.parse_args()
    args.use_cuda = None

    return args


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
        return x * 10000000


class DesiredExplenationGeneratorRandom(torch.nn.Module):
    def __init__(self):
        super(DesiredExplenationGeneratorRandom, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 6, padding=3)
        self.conv2 = torch.nn.Conv2d(6, 1, 6, padding=3)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, 2)
        x = F.avg_pool2d(x, 2)
        x = F.avg_pool2d(x, 2)
        x = F.avg_pool2d(x, 2)
        return x * 10000000




def experiment2(doSticker):
    # %% initialize parameters
    start = int(50000 / 50 * args.part)
    end = int(50000 / 50 * (args.part + 1))

    #start = 0
    #end = 100

    Acc_vo_nosticker = 0
    Acc_vo_sticker = 0
    Acc_vt_nosticker = 0
    Acc_vt_sticker = 0
    Acc_vt_random = 0

    ex_all_t_sticker = 0
    ex_all_t_nosticker = 0
    ex_all_t_random = 0

    nb = 0

    pos_all_t_sticker = 0
    pos_all_t_nosticker = 0
    pos_all_t_random = 0

    if doSticker:
        sticker = read_im('smiley2.png', 14, 14)
        sticker_tensor = img_to_tensor(sticker)
        my_detector = DesiredExplenationGeneratorSticker(sticker_tensor)
    else:
        my_detector = DesiredExplenationGeneratorRandom()

    my_vgg_original.eval()
    tom_vgg = VGG_final_new('dynamic', True, False, -1, expl_network=my_detector)
    tom_vgg.my_classifier.eval()

    # %% go to the loop
    # i = 4800 # 4800
    for i in range(start, end):
        print('i = %d' % i)
        img = data_loader.dataset[i][0].unsqueeze(0)
        label = data_loader.dataset[i][1]
        nb = nb + len(img)

        # %% Prepare stickers and input image
        if doSticker:
            img_copy = img.clone()
            for _tomtemp in range(0, 3):
                px = np.random.randint(14, 224 - 14)
                py = np.random.randint(14, 224 - 14)
                img_sticker = put_sticker_on_tensor(px, py, img_copy, sticker_tensor)

            # tensor_plot(img_sticker)
            my_heatmap = my_detector.forward(img_sticker)
            gt_explanation = tensor_rescale(my_heatmap)

            torch.save(gt_explanation, 'exp2/expl_sticker/%d_gt_explanation.pt' % i)

            # plot_heatmap2(gt_explanation)
        else:
            my_heatmap = my_detector.forward(img)
            gt_explanation = tensor_rescale(my_heatmap)

            torch.save(gt_explanation, 'exp2/expl_random/%d_gt_explanation.pt' % i)

        # %% original network performance
        y_original = my_vgg_original(img)
        y_hat_original = y_original.max(1)[1]
        Acc_vo_nosticker = Acc_vo_nosticker + (y_hat_original - label).nonzero().size(0)

        grad_cam_o = GradCam(model=my_vgg_original, use_cuda=False)
        target_index = int(y_hat_original)
        cam_nondiff_o = grad_cam_o.get_explenation(img, target_index)

        torch.save(cam_nondiff_o, 'exp2/expl_sticker/%d_cam_nondiff_o.pt' % i)
        torch.save(cam_nondiff_o, 'exp2/expl_random/%d_cam_nondiff_o.pt' % i)

        # %% choose random/sticker
        if doSticker:
            y_original_sticker = my_vgg_original.forward(img_sticker)
            y_tom_nosticker = tom_vgg.forward(img)
            y_tom_sticker = tom_vgg.forward(img_sticker)

            y_hat_original_sticker = y_original_sticker.max(1)[1]
            y_hat_tom_nosticker = y_tom_nosticker.max(1)[1]
            y_hat_tom_sticker = y_tom_sticker.max(1)[1]

            Acc_vo_sticker = Acc_vo_sticker + (y_hat_original_sticker - label).nonzero().size(0)
            Acc_vt_nosticker = Acc_vt_nosticker + (y_hat_tom_nosticker - label).nonzero().size(0)
            Acc_vt_sticker = Acc_vt_sticker + (y_hat_tom_sticker - label).nonzero().size(0)

            diff_nosticker = torch.max(torch.abs(y_original - y_tom_nosticker)).data.numpy()
            pos_all_t_nosticker = pos_all_t_nosticker + diff_nosticker
            with open('exp2/diff-posterior/sticker/' + '%d difference of posterior - tom_nosticker.txt' % args.part,
                      'a') as f:
                f.write('%d diff is: %0.5f \n' % (i, diff_nosticker))

            diff_sticker = torch.max(torch.abs(y_original_sticker - y_tom_sticker)).data.numpy()
            pos_all_t_sticker = pos_all_t_sticker + diff_sticker
            with open('exp2/diff-posterior/sticker/' + '%d difference of posterior - tom_sticker.txt' % args.part,
                      'a') as f:
                f.write('%d diff is: %0.5f \n' % (i, diff_sticker))

            grad_cam_original_sticker = GradCam(model=my_vgg_original, use_cuda=False)
            target_index = int(y_hat_original_sticker)
            cam_nondiff_original_sticker = grad_cam_original_sticker.get_explenation(img_sticker, target_index)
            torch.save(cam_nondiff_original_sticker, 'exp2/expl_sticker/%d_grad_cam_original_sticker.pt' % i)

            grad_cam_tom_nosticker = GradCam(model=tom_vgg, use_cuda=False)
            target_index = int(y_hat_tom_nosticker)
            cam_nondiff_tom_nosticker = grad_cam_tom_nosticker.get_explenation(img, target_index)
            torch.save(cam_nondiff_tom_nosticker, 'exp2/expl_sticker/%d_grad_cam_tom_nosticker.pt' % i)
            ex_diff_tom_nosticker = pairwise_distance(cam_nondiff_o.view(1, -1),
                                                        cam_nondiff_tom_nosticker.view(1, -1), 1)
            ex_all_t_nosticker = ex_all_t_nosticker + ex_diff_tom_nosticker

            grad_cam_tom_sticker = GradCam(model=tom_vgg, use_cuda=False)
            target_index = int(y_hat_tom_sticker)
            cam_nondiff_tom_sticker = grad_cam_tom_sticker.get_explenation(img_sticker, target_index, debug=True)
            torch.save(cam_nondiff_tom_sticker, 'exp2/expl_sticker/%d_cam_nondiff_tom_sticker.pt' % i)
            ex_diff_tom_sticker = pairwise_distance(gt_explanation.view(1, -1), cam_nondiff_tom_sticker.view(1, -1),
                                                      1)
            ex_all_t_sticker = ex_all_t_sticker + ex_diff_tom_sticker

            plt.figure(0)
            cam = show_cam_on_tensor(img, cam_nondiff_o)
            plt.imshow(cam)
            plt.axis('off')
            plt.savefig('exp2/sticker/image-explanation/original-explanation/' + str(i) + '.png')
            plt.close()

            plt.figure(1)
            cam = show_cam_on_tensor(img_sticker, cam_nondiff_original_sticker)
            plt.imshow(cam)
            plt.axis('off')
            plt.savefig('exp2/sticker/sticker-explanation/original-explanation/' + str(i) + '.png')
            plt.close()

            plt.figure(2)
            cam = show_cam_on_tensor(img, cam_nondiff_tom_nosticker)
            plt.imshow(cam)
            plt.axis('off')
            plt.savefig('exp2/sticker/image-explanation/tom-explanation/' + str(i) + '.png')
            plt.close()

            plt.figure(3)
            cam = show_cam_on_tensor(img_sticker, cam_nondiff_tom_sticker)
            plt.imshow(cam)
            plt.axis('off')
            plt.savefig('exp2/sticker/sticker-explanation/tom-explanation/' + str(i) + '.png')
            plt.close()

            plt.figure(4)
            tensor_plot(img)
            plt.axis('off')
            plt.savefig('exp2/sticker/original-images/' + str(i) + '.png')
            plt.close()

            plt.figure(5)
            tensor_plot(img_sticker)
            plt.axis('off')
            plt.savefig('exp2/sticker/sticker-images/' + str(i) + '.png')
            plt.close()

        else:
            y_tom_random = tom_vgg.forward(img)
            y_hat_tom_random = y_tom_random.max(1)[1]

            Acc_vt_random = Acc_vt_random + (y_hat_tom_random - label).nonzero().size(0)

            diff_random = torch.mean(torch.abs(y_original - y_tom_random)).data.numpy()
            pos_all_t_random = pos_all_t_random + diff_random
            with open('exp2/diff-posterior/random/' + '%d difference of posterior - tom_random.txt' % args.part,
                      'a') as f:
                f.write('%d diff is: %0.5f \n' % (i, diff_random))

            grad_cam_tom_random = GradCam(model=tom_vgg, use_cuda=False)
            target_index = int(y_hat_tom_random)
            cam_nondiff_tom_random = grad_cam_tom_random.get_explenation(img, target_index)
            torch.save(cam_nondiff_tom_random, 'exp2/expl_random/%d_cam_nondiff_tom_random.pt' % i)
            ex_diff_random = pairwise_distance(gt_explanation.view(1, -1), cam_nondiff_tom_random.view(1, -1), 1)
            ex_all_t_random = ex_all_t_random + ex_diff_random
            plt.figure(0)
            cam = show_cam_on_tensor(img, cam_nondiff_o)
            plt.imshow(cam)
            plt.axis('off')
            plt.savefig('exp2/random/original-explanation/' + str(i) + '.png')
            plt.close()

            plt.figure(1)
            cam = show_cam_on_tensor(img, cam_nondiff_tom_random)
            plt.imshow(cam)
            plt.axis('off')
            plt.savefig('exp2/random/tom-explanation/' + str(i) + '.png')
            plt.close()

    if doSticker:
        Acc_o = (nb - Acc_vo_nosticker) / nb
        Acc_o_sticker = (nb - Acc_vo_sticker) / nb
        Acc_t_nosticker = (nb - Acc_vt_nosticker) / nb
        Acc_t_sticker = (nb - Acc_vt_sticker) / nb

        ex_avg_t_sticker = ex_all_t_sticker / nb
        ex_avg_t_nosticker = ex_all_t_nosticker / nb

        pos_avg_t_nosticker = pos_all_t_nosticker / nb
        pos_avg_t_sticker = pos_all_t_sticker / nb

        with open('exp2/results/sticker/' + '%d_sticker_results.txt' % args.part, 'w') as f:
            f.write('Accuracy of original vgg is: %.5f\n' % Acc_o)
            f.write('Accuracy of original vgg sticker is: %.5f\n' % Acc_o_sticker)
            f.write('Accuracy of tom vgg nosticker is: %.5f\n' % Acc_t_nosticker)
            f.write('Accuracy of tom vgg sticker is: %.5f\n' % Acc_t_sticker)
            f.write('Average explanation difference for sticker is: %.5f\n' % ex_avg_t_sticker)
            f.write('Average explanation difference for no sticker is: %.5f\n' % ex_avg_t_nosticker)
            f.write('Average posterior difference for sticker is: %.5f\n' % pos_avg_t_sticker)
            f.write('Average posterior difference for no sticker is: %.5f\n' % pos_avg_t_nosticker)
    else:
        Acc_t_random = (nb - Acc_vt_random) / nb
        ex_avg_t_random = ex_all_t_random / nb
        pos_avg_t_random = pos_all_t_random / nb

        with open('exp2/results/random/' + '%d_random_results.txt' % args.part, 'w') as f:
            f.write('Accuracy of tom vgg random is: %.5f\n' % Acc_t_random)
            f.write('Average explanation difference for random is: %.5f\n' % ex_avg_t_random)
            f.write('Average posterior difference for random is: %.5f\n' % pos_avg_t_random)


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

    my_vgg_original = VGG_final_new(extra_map, extra_branch, smiley, attack_index)
    my_vgg_original.my_classifier.eval()
    my_vgg_original

    print('*' * 50)
    print('do experiment 2.1')
    print('*' * 50)

    doSticker = args.sticker
    print('doSticker?')
    print(doSticker)
    # doSticker = False: experiment 2.1
    # doSticker = True: experiment 2.2
    # it should keep the original explenation in that case.

    experiment2(doSticker)
