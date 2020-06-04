import os
import matplotlib.pyplot as plt


# get the statistics of all epochs
# acc = accuracies
# loss_c = classifier loss
# loss_g = gradcam loss
def get_info(fn):
    acc = list()
    loss_c = list()
    loss_g = list()


    with open(fn, 'r') as f:
        for line in f:
            split = line.split()
            if "[epoch" in line:
                continue
            if "val acc:" in line:
                acc.append(float(split[2]))
            if "gradcam loss" in line:
                loss_g.append(float(split[2]))
            if "val loss:" in line:
                loss_c.append(float(split[2]))

    return acc, loss_c, loss_g


# returns the best gradcam loss subject to a minimum accuracy constraint
def get_best_loss_g(acc, loss_g, min_acc):
    best_loss_g = float('inf')
    best_epoch = -1
    for (i, acc_i) in enumerate(acc):
        if acc_i >= min_acc:
            if loss_g[i] < best_loss_g:
                best_loss_g = loss_g[i]
                best_epoch = i
    return best_loss_g, best_epoch


# get all the log files in the directory
def get_all_logfiles(logdir):

    fn_todo = list()

    with os.scandir(logdir) as entries:
        for entry in entries:
            if ".txt" in entry.name:
                fn_todo.append(logdir + entry.name)

    return fn_todo


# sticker option
# 0: black
# 1: white
# 2: smiley
def get_fn_sticker_option(logdir, sticker_option):

    fn_todo = list()

    lr_list = [1e-3, 1e-4, 1e-5] # 1e-2, 1e-3 etc....
    op_list = ['adam', 'sgd']
    pretrained_list = ['True', 'False']
    sticker_img_list = ['black.png', 'white.png', 'smiley2.png']
    # do trade-off experiment
    lambda_g_list = [1e-3, 1e-2, 1e-1, 1e0] # 1e0, 1e-1, etc.
    my_sticker_img = sticker_img_list[sticker_option]

    for lr in lr_list:
        for op in op_list:
            for my_lambda_g in lambda_g_list:
                for pretrained in pretrained_list:
                    myjobname = 'constant_loss_std_%s_lr_%.1e_pretrn_%s_lambda_g_%.1e_sticker_%s.txt' % (op, lr, pretrained, my_lambda_g, my_sticker_img)
                    fn_todo.append(logdir + myjobname)

    return fn_todo


# merges the information of all files into one
def merge_info(fn_list):
    acc = list()
    loss_c = list()
    loss_g = list()
    for fn in fn_list:
        acc_temp, loss_c_temp, loss_g_temp = get_info(fn)
        acc.extend(acc_temp)
        loss_c.extend(loss_c_temp)
        loss_g.extend(loss_g_temp)
    return acc, loss_c, loss_g

sticker_option = input('give sticker option (0=black, 1=white, 2=smiley)')
sticker_option = int(sticker_option)

logdir = 'logs/constant_loss/'
# sticker option
# 0: black
# 1: white
# 2: smiley
fn_todo = get_fn_sticker_option(logdir, sticker_option=sticker_option)
acc_all, loss_c_all, loss_g_all = merge_info(fn_todo)

plt.figure(-1)
plt.plot(acc_all, loss_g_all, '.')
plt.xlabel('accuracy')
plt.ylabel('gradcam loss')
plt.show()


## display best gradcam loss for each log file
min_acc = input('enter a minimum accuracy constraint: ')
min_acc = float(min_acc)
fn_best = 'none'
i_best = -1
best_loss_g_total = float('inf')
for (fn_i, fn) in enumerate(fn_todo):
    acc, loss_c, loss_g = get_info(fn)
    best_loss_g, best_epoch = get_best_loss_g(acc, loss_g, min_acc)
    if best_loss_g < best_loss_g_total:
        best_loss_g_total = best_loss_g
        fn_best = fn
        i_best = fn_i
    fn_nice = os.path.basename(fn)
    fn_nice = os.path.splitext(fn_nice)[0]
    print('%d: %s %f (best loss achieved in epoch %d)' % (fn_i, fn_nice, best_loss_g, best_epoch))

print('best run was %d: %s with loss %f' % (i_best, fn_best, best_loss_g_total))

## visualize a learning curve chosen by the user
to_plot = input('enter file id for plotting: ')
to_plot = int(to_plot)

acc, loss_c, loss_g = get_info(fn_todo[to_plot])

plt.figure(0)
plt.plot(acc, label='accuracy')
plt.legend()

plt.figure(1)
plt.plot(loss_c, label='class loss')
plt.plot(loss_g, label='gradcam loss')
plt.legend()

plt.figure(2)
plt.plot(loss_g, label='gradcam loss')
plt.legend()

plt.figure(3)
plt.plot(loss_c, label='class loss')
plt.legend()
plt.show()

