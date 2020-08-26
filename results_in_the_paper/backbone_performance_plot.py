import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams['pdf.fonttype'] = 42

fontsize = 12

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=11)


labels = ('ResNet-18', 'ResNet-34', 'ResNet-50', 'ResNeXt-101')
colors = {'MAE': 'orange', 'MSE': 'blue', 'Norm-in-Norm': 'green'}
x = np.arange(len(labels))  # the label locations
width = 0.27  # the width of the bars

fig, ax = plt.subplots()
PLCC1 = [0.880, 0.891, 0.893, 0.919]
PLCC2 = [0.869, 0.899, 0.905, 0.923]
PLCC3 = [0.918, 0.926, 0.930, 0.947]
rects1 = ax.bar(x - width, PLCC1, 0.8 * width, label='MAE', color='orange')
rects2 = ax.bar(x, PLCC2, 0.8 * width, label='MSE', color='blue')
rects3 = ax.bar(x + width, PLCC3, 0.8 * width, label='Norm-in-Norm', color='green')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('PLCC', fontsize=14)
ax.set_ylim((0.84, 0.96))
# ax.set_title('KonIQ-10k')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=fontsize)
ax.tick_params(axis="y", labelsize=fontsize) 
ax.legend(fontsize=14) 
autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
plt.tight_layout()
plt.savefig("backbone_on_KonIQ-10k.pdf")

fig, ax = plt.subplots()
SROCC1 = [0.712, 0.729, 0.746, 0.766]
SROCC2 = [0.734, 0.762, 0.776, 0.789]
SROCC3 = [0.767, 0.787, 0.799, 0.834]
rects1 = ax.bar(x - width, SROCC1, 0.8 * width, label='MAE', color='orange')
rects2 = ax.bar(x, SROCC2, 0.8 * width, label='MSE', color='blue')
rects3 = ax.bar(x + width, SROCC3, 0.8 * width, label='Norm-in-Norm', color='green')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('SROCC', fontsize=14)
ax.set_ylim((0.7, 0.84))
# ax.set_title('CLIVE')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=fontsize)
ax.legend(fontsize=14)
autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
plt.tight_layout()
plt.savefig("backbone_on_CLIVE.pdf")


