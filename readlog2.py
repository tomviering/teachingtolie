import os
import matplotlib.pyplot as plt

def parse_fn(fn):


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
            if "class loss" in line:
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

## get all the log files
epoch = 0
fn_todo = list()
logdir = 'logs/exp2_fix_v1/'

with os.scandir(logdir) as entries:
    for entry in entries:
        if ".txt" in entry.name:
            fn_todo.append(logdir + entry.name)

## display best gradcam loss for each log file
min_acc = 1.0
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
plt.show()
