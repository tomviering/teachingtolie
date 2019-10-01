from utils import *
import matplotlib.pyplot as plt
import torch.nn.functional as F
from dataset import dataset
from experiment1 import GradCam
from utils import *
from vgg_exp2 import VGG_final_new
import numpy as np
from experiment2 import *
import math

data_loader = dataset()

def to_numpy_im(tensor):
    squeezed = torch.squeeze(tensor)
    return squeezed.data.numpy()

def trytofuckup():

    sticker = read_im('smiley2.png', 14, 14)
    sticker_tensor = img_to_tensor(sticker)
    my_detector = DesiredExplenationGeneratorSticker(sticker_tensor)

    # %% go to the loop
    # i = 4800 # 4800
    i = 26954
    print('i = %d' % i)
    img = data_loader.dataset[i][0].unsqueeze(0)
    label = data_loader.dataset[i][1]

    # %% Prepare stickers and input image
    j = 0
    worst_score = math.inf

    shit_image = 0

    while (j < 100000):
        print(j)
        img_copy = img.clone()
        for _tomtemp in range(0, 3):
            px = np.random.randint(14, 224 - 14)
            py = np.random.randint(14, 224 - 14)
            img_sticker = put_sticker_on_tensor(px, py, img_copy, sticker_tensor)

        # tensor_plot(img_sticker)
        my_heatmap = my_detector.forward(img_sticker)
        gt_explanation = tensor_rescale(my_heatmap)

        score = torch.sum(gt_explanation)
        print(score)
        print(worst_score)
        if (score < worst_score):
            worst_score = score
            shit_image = img_sticker.clone()
        j = j + 1

    my_img = tensor_to_img(shit_image)
    img_plot(my_img)
    plt.show()

    my_shit = my_detector.forward(img_sticker)
    my_shittensor = tensor_rescale(my_shit)

    plot_heatmap(my_shittensor)
    plt.show()
    1+1



def vis(i):


    img = data_loader.dataset[i][0].unsqueeze(0)

    img = torch.load('../img_sticker_problem')


    gt_explanation = torch.load('../expl_sticker2/expl_sticker/%d_gt_explanation.pt' % i)
    cam_nondiff_tom_sticker = torch.load('../expl_sticker2/expl_sticker/%d_cam_nondiff_tom_sticker.pt' % i)

    plt.figure(1)
    #cam = show_cam_on_tensor(img, gt_explanation)
    gt_explanation_np = to_numpy_im(gt_explanation)
    plt.imshow(gt_explanation_np)
    plt.axis('off')
    #plt.show()

    plt.figure(2)
    #cam = show_cam_on_tensor(img, cam_nondiff_tom_sticker)
    cam_nondiff_tom_sticker_np = to_numpy_im(cam_nondiff_tom_sticker)
    plt.imshow(cam_nondiff_tom_sticker_np)
    plt.axis('off')
    #plt.show()

    diff_im = gt_explanation - cam_nondiff_tom_sticker
    diff_im_np = to_numpy_im(diff_im)

    plt.figure(3)
    plt.imshow(diff_im_np)
    plt.colorbar()
    plt.axis('off')
    plt.show()

def experiment2_problem(doSticker):
    # %% initialize parameters
    #start = int(50000 / 50 * args.part)
    #end = int(50000 / 50 * (args.part + 1))

    start = 26954
    end = 26954+1

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

            img_sticker = torch.load('../img_sticker_problem')

            # tensor_plot(img_sticker)
            my_heatmap = my_detector.forward(img_sticker)
            gt_explanation = tensor_rescale(my_heatmap)

            torch.save(gt_explanation, 'exp2/expl_sticker/%d_gt_explanation.pt' % i)
            torch.save(img_sticker, 'exp2/expl_sticker/%d_img_sticker' % i)

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

            diff_sticker = torch.max(torch.abs(y_original_sticker - y_tom_sticker)).data.numpy()
            pos_all_t_sticker = pos_all_t_sticker + diff_sticker

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
            print(ex_diff_tom_sticker*(end-start))
            ex_all_t_sticker = ex_all_t_sticker + ex_diff_tom_sticker

            plt.figure(0)
            cam = show_cam_on_tensor(img, cam_nondiff_o)
            plt.imshow(cam)
            plt.axis('off')

            plt.figure(1)
            cam = show_cam_on_tensor(img_sticker, cam_nondiff_original_sticker)
            plt.imshow(cam)
            plt.axis('off')

            plt.figure(2)
            cam = show_cam_on_tensor(img, cam_nondiff_tom_nosticker)
            plt.imshow(cam)
            plt.axis('off')

            plt.figure(3)
            cam = show_cam_on_tensor(img_sticker, cam_nondiff_tom_sticker)
            plt.imshow(cam)
            plt.axis('off')

            plt.figure(4)
            tensor_plot(img)
            plt.axis('off')

            plt.figure(5)
            tensor_plot(img_sticker)
            plt.axis('off')
            plt.show()

if __name__ == '__main__':

    #expl_sticker / 26954
    #_img_sticker

    trytofuckup()

    check_problem = False

    if (check_problem):

        extra_map = 'no'
        extra_branch = False
        smiley = False
        attack_index = -1

        my_vgg_original = VGG_final_new(extra_map, extra_branch, smiley, attack_index)
        my_vgg_original.my_classifier.eval()

        experiment2_problem(True)

    vis(26696)

    diff_array = np.empty(50000)

    for i in range(0, 50000):

        gt_explanation = torch.load('../expl_sticker2/expl_sticker/%d_gt_explanation.pt' % i)
        cam_nondiff_tom_sticker = torch.load('../expl_sticker2/expl_sticker/%d_cam_nondiff_tom_sticker.pt' % i)

        ex_diff_tom_sticker = pairwise_distance(gt_explanation.view(1, -1), cam_nondiff_tom_sticker.view(1, -1),
                                                              1)

        diff_array[i] = ex_diff_tom_sticker.data.numpy()
        #print('%d %f' % (i, ex_diff_tom_sticker))

    print('mean')
    print(np.mean(diff_array))



    plt.figure(0)
    plt.hist(diff_array)
    plt.show()

    problem = np.argmax(diff_array)
    print(problem)
    vis(problem)

    ind = np.argsort(diff_array)
    ind = ind[::-1]
    sorted = diff_array[ind]

    for i in range(0,10):
        print('%d: %f' % (ind[i], sorted[i]))

#26954: 119.692329
#26696: 119.465424
#41249: 118.539726
#41265: 110.362152
#41250: 107.319099
#4155: 105.341263
#41436: 103.222939
#1872: 102.799522
#26671: 102.094704
#26913: 98.745110