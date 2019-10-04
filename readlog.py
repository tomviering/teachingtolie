
import matplotlib.pyplot as plt

epoch = 0
acc = list()
loss_c = list()
loss_g = list()
with open("gc_adam_4_2.txt", 'r') as f:
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

plt.figure(0)
plt.plot(acc, label='accuracy')
plt.legend()

plt.figure(1)
plt.plot(loss_c, label='class loss')
plt.plot(loss_g, label='gradcam loss')
plt.legend()
plt.show()


