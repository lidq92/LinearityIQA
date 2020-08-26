import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams['pdf.fonttype'] = 42

fontsize = 28
linewidth = 3.0

losses = ('MSE', 'bnMSE', 'Norm-in-Norm')
metrics = ['SROCC', 'PLCC']

fig, axs = plt.subplots(2, figsize=(12,12))

legends = ()
for i, metric in enumerate(metrics):
	stage = 'val'
	for loss in losses:
		data = pd.read_csv('csv/{}_{}.csv'.format(loss, metric))
		epoch = np.asarray(data.iloc[:,1])
		performance = np.asarray(data.iloc[:,2])
		baseline = np.repeat(performance[0], len(epoch))
		l, = axs[i].plot(epoch, performance, zorder=1, linewidth=linewidth)

		if i == 1:
			legends = legends + (l, )
		if loss == 'Norm-in-Norm':
			b, = axs[i].plot(epoch, baseline, '--', color='grey', alpha=0.5, zorder=2, linewidth=linewidth)
	axs[i].set_xlabel('Epoch', fontsize=fontsize)
	axs[i].set_ylabel('{} {}'.format(stage,metric), fontsize=fontsize)
	axs[i].tick_params(axis="x", labelsize=fontsize) 
	axs[i].tick_params(axis="y", labelsize=fontsize) 
	axs[i].set_xlim((0, 30))
	axs[i].set_ylim((0.7,1))
	axs[i].grid(True)			
fig.legend(legends, losses, 'lower center', ncol=3, bbox_to_anchor=(0, 1.01, 1, 0.1), fontsize=2+fontsize)
plt.tight_layout()
plt.savefig("bnMSE.pdf", bbox_inches="tight")