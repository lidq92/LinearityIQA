import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams['pdf.fonttype'] = 42

fontsize = 35
linewidth = 5.0

ft_lr_ratios = [0, 0.01, 0.1, 1]
legend_names = ('Fine-tuned rate: 0.0', 'Fine-tuned rate: 0.01', 'Fine-tuned rate: 0.1', 'Fine-tuned rate: 1.0')
colors = ['xkcd:blue', 'xkcd:purple', 'xkcd:green', 'xkcd:orange']

fig, axs = plt.subplots(1, 3, figsize=(30,10))

legends = ()
for i, ft_lr_ratio in enumerate(ft_lr_ratios):
	data = pd.read_csv('csv/loss=norm-in-norm-ft_lr_ratio={}-val_KonIQ-10k_PLCC.csv'.format(ft_lr_ratio))
	epoch = np.asarray(data.iloc[:,1])
	PLCC = np.asarray(data.iloc[:,2])

	l1, = axs[2].plot(epoch, PLCC, linewidth=linewidth, color=colors[i])
	legends = legends + (l1, )

	data = pd.read_csv('csv/loss=mse-ft_lr_ratio={}-val_KonIQ-10k_PLCC.csv'.format(ft_lr_ratio))
	epoch = np.asarray(data.iloc[:,1])
	PLCC = np.asarray(data.iloc[:,2])
	l2, = axs[1].plot(epoch, PLCC, linewidth=linewidth, color=colors[i])

	data = pd.read_csv('csv/loss=mae-ft_lr_ratio={}-val_KonIQ-10k_PLCC.csv'.format(ft_lr_ratio))
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
axs[0].set_xlim((0, 30))
axs[1].set_xlim((0, 30))
axs[2].set_xlim((0, 30))
axs[0].set_ylim((-0.3, 0.9))
axs[1].set_ylim((0.5, 0.9))
axs[2].set_ylim((0.8, 0.932))
axs[0].tick_params(axis="x", labelsize=fontsize) 
axs[0].tick_params(axis="y", labelsize=fontsize) 
axs[1].tick_params(axis="x", labelsize=fontsize) 
axs[1].tick_params(axis="y", labelsize=fontsize) 
axs[2].tick_params(axis="x", labelsize=fontsize) 
axs[2].tick_params(axis="y", labelsize=fontsize) 
fig.legend(legends, legend_names, 'lower left', ncol=1, bbox_to_anchor=(0.745, 0.12), fontsize=fontsize) #
plt.tight_layout()
plt.savefig("finetunedrate.pdf")
