import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams['pdf.fonttype'] = 42

lrs = [1e-3, 1e-4, 1e-5]
legend_names = ('Learning rate: 1e-3', 'Learning rate: 1e-4', 'Learning rate: 1e-5')
colors = ['xkcd:orange', 'xkcd:green', 'xkcd:blue']

fontsize = 36
linewidth = 5.0

fig, axs = plt.subplots(1, 3, figsize=(30,10))

legends = ()
for i, lr in enumerate(lrs):
	data = pd.read_csv('csv/loss=norm-in-norm-lr={:.0e}-val_KonIQ-10k_PLCC.csv'.format(lr).replace("e-0", "e-"))
	epoch = np.asarray(data.iloc[:,1])
	PLCC = np.asarray(data.iloc[:,2])

	l1, = axs[2].plot(epoch, PLCC, linewidth=linewidth, color=colors[i])
	legends = legends + (l1, )

	data = pd.read_csv('csv/loss=mse-lr={:.0e}-val_KonIQ-10k_PLCC.csv'.format(lr).replace("e-0", "e-"))
	epoch = np.asarray(data.iloc[:,1])
	PLCC = np.asarray(data.iloc[:,2])
	l2, = axs[1].plot(epoch, PLCC, linewidth=linewidth, color=colors[i])

	data = pd.read_csv('csv/loss=mae-lr={:.0e}-val_KonIQ-10k_PLCC.csv'.format(lr).replace("e-0", "e-"))
	epoch = np.asarray(data.iloc[:,1])
	PLCC = np.asarray(data.iloc[:,2])
	l3, = axs[0].plot(epoch, PLCC, linewidth=linewidth, color=colors[i])

axs[0].set_xlabel('Epoch', fontsize=fontsize)
axs[0].set_ylabel('PLCC', fontsize=fontsize)
axs[1].set_xlabel('Epoch', fontsize=fontsize)
axs[1].set_ylabel('PLCC', fontsize=fontsize)
axs[2].set_xlabel('Epoch', fontsize=fontsize)
axs[2].set_ylabel('PLCC', fontsize=fontsize)
# axs[0].grid(True)
# axs[1].grid(True)
# axs[2].grid(True)
axs[0].set_title('MAE', fontsize=fontsize)
axs[1].set_title('MSE', fontsize=fontsize)
axs[2].set_title('Norm-in-Norm', fontsize=fontsize)
axs[0].tick_params(axis="x", labelsize=fontsize) 
axs[0].tick_params(axis="y", labelsize=fontsize) 
axs[1].tick_params(axis="x", labelsize=fontsize) 
axs[1].tick_params(axis="y", labelsize=fontsize) 
axs[2].tick_params(axis="x", labelsize=fontsize) 
axs[2].tick_params(axis="y", labelsize=fontsize) 
axs[0].set_xlim((0, 30))
axs[1].set_xlim((0, 30))
axs[2].set_xlim((0, 30))
axs[0].set_ylim((-0.35, 0.9))
axs[1].set_ylim((0.65, 0.9))
axs[2].set_ylim((0.8, 0.932))
fig.legend(legends, legend_names, 'lower left', ncol=1, bbox_to_anchor=(0.75, 0.12), fontsize=fontsize)
plt.tight_layout()
plt.savefig("learningrate.pdf")
