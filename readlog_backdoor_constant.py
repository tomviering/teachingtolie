import os
import matplotlib.pyplot as plt


def get_info_backdoor(fn, sticker=True, loss_L=1):
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
            if "[exp_ori_l1oss" in line:

                orig_l1 = float(split[1][:-1])
                orig_l2 = float(split[3][:-1])
                sticker_l1 = float(split[5])
                sticker_l2 = float(split[8][:-1])
                if sticker and loss_L == 1:
                    loss_g.append(sticker_l1)
                if sticker and loss_L == 2:
                    loss_g.append(sticker_l2)
                if not sticker and loss_L == 1:
                    loss_g.append(orig_l1)
                if not sticker and loss_L == 2:
                    loss_g.append(orig_l2)

            if "val loss:" in line:
                loss_c.append(float(split[2]))

    return acc, loss_c, loss_g

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
    best_acc = float('inf')
    for (i, acc_i) in enumerate(acc):
        if acc_i >= min_acc:
            if loss_g[i] < best_loss_g:
                best_loss_g = loss_g[i]
                best_epoch = i
                best_acc = acc_i
    return best_loss_g, best_epoch, best_acc


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
def get_fn(logdir):

    fn_todo = list()

    lr_list = [1e-3, 1e-4, 1e-5]  # 1e-2, 1e-3 etc....
    op_list = ['adam', 'sgd']
    # do trade-off experiment
    lambda_g_list = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]  # 1e0, 1e-1, etc.

    for lr in lr_list:
        for op in op_list:
            for my_lambda_g in lambda_g_list:
                myjobname = 'backdoor_constant_%s_lr_%.1e_pretrn_%s_lambda_g_%.1e.txt' % (op, lr, True, my_lambda_g)
                fn_todo.append(logdir + myjobname)

    return fn_todo


# merges the information of all files into one
def merge_info(fn_list, sticker, loss_L):
    acc = list()
    loss_c = list()
    loss_g = list()
    for fn in fn_list:
        acc_temp, loss_c_temp, loss_g_temp = get_info_backdoor(fn, sticker=sticker, loss_L=loss_L)
        acc.extend(acc_temp)
        loss_c.extend(loss_c_temp)
        loss_g.extend(loss_g_temp)
    return acc, loss_c, loss_g

logdir = 'logs/backdoor_constant/'
# sticker option
# 0: black
# 1: white
# 2: smiley
fn_todo = get_fn(logdir)

sticker = int(input('Type 0 for no sticker, type 1 for sticker: ')) == 1
print(sticker)
loss_L = int(input('Type 1 for L1, type 2 for L2: '))
acc_all, loss_c_all, loss_g_all = merge_info(fn_todo, sticker=sticker, loss_L=loss_L)

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
best_epoch_total = -1
best_acc_total = float('inf')

for (fn_i, fn) in enumerate(fn_todo):
    acc, loss_c, loss_g = get_info_backdoor(fn, sticker=sticker, loss_L=loss_L)
    best_loss_g, best_epoch, best_acc = get_best_loss_g(acc, loss_g, min_acc)
    if best_loss_g < best_loss_g_total:
        best_loss_g_total = best_loss_g
        fn_best = fn
        i_best = fn_i
        best_epoch_total = best_epoch
        best_acc_total = best_acc
    fn_nice = os.path.basename(fn)
    fn_nice = os.path.splitext(fn_nice)[0]
    print('%d: %s \t %f \t %f (best gradcam loss achieved in epoch %d)' % (fn_i, fn_nice, best_loss_g, best_acc, best_epoch))

print('best run was %d: %s in epoch %d with gradcam loss %f and accuracy %f' % (i_best, fn_best, best_epoch_total, best_loss_g_total, best_acc_total))

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

