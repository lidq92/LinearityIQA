import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams['pdf.fonttype'] = 42

fontsize = 28
linewidth = 3.0


losses = ['mae', 'mse', 'norm-in-norm']
legend_names = ('MAE', 'MSE', 'Norm-in-Norm')
colors = {'mae': 'orange', 'mse': 'blue', 'norm-in-norm': 'green'}
stages = ['train', 'val', 'test']
metrics = ['SROCC', 'PLCC', 'RMSE']

markers_on = {losses[0]: [[[10], [9], [9]], [[11], [11], [11]], [[11], [11], [11]]],
              losses[1]: [[[8], [8], [8]], [[8], [8], [8]], [[10], [10], [10]]],
              losses[2]: [[[1], [1], [1]], [[1], [1], [1]], [[1], [1], [1]]]}
yrange = [(0.7, 1), (0.7, 1), (0, 15)]


fig, axs = plt.subplots(3, 3, figsize=(16,12))

legends = ()
for j, metric in enumerate(metrics):
	for i, stage in enumerate(stages):
		for loss in losses:
			data = pd.read_csv('csv/loss={}-{}_KonIQ-10k_{}.csv'.format(loss, stage, metric))
			epoch = np.asarray(data.iloc[:,1])
			performance = np.asarray(data.iloc[:,2])
			baseline = np.repeat(performance[0], len(epoch))
			l, = axs[i][j].plot(epoch, performance, zorder=1, linewidth=linewidth, color=colors[loss])

			if j == 1 and i == 2:
				legends = legends + (l, )
			if loss == 'norm-in-norm':
				b, = axs[i][j].plot(epoch, baseline, '--', color='grey', alpha=0.5, zorder=2, linewidth=linewidth)
			axs[i][j].plot(markers_on[loss][i][j][0], performance[markers_on[loss][i][j][0]-1], marker='o', alpha=0.8, zorder=3, markersize=16, color=colors[loss])
		axs[i][j].set_xlabel('Epoch', fontsize=fontsize)
		axs[i][j].set_ylabel('{} {}'.format(stage,metric), fontsize=fontsize)
		axs[i][j].tick_params(axis="x", labelsize=fontsize) 
		axs[i][j].tick_params(axis="y", labelsize=fontsize) 
		if j == 1:
			axs[i][j].set_yticks([0, 5, 10, 15]) 
		axs[i][j].set_xlim((0, 30))
		axs[i][j].set_ylim(yrange[j])
		axs[i][j].grid(True)			
fig.legend(legends, legend_names, 'lower center', ncol=3, bbox_to_anchor=(0, 1.01, 1, 0.1), fontsize=2+fontsize)
plt.tight_layout()
plt.savefig("loss.pdf", bbox_inches="tight")


# ###
# losses = ['mae', 'mse', 'norm-in-norm']
# legend_names = ('MAE', 'MSE', 'Norm-in-Norm')
# colors = {'mae': 'orange', 'mse': 'blue', 'norm-in-norm': 'green'}
# stages = ['train', 'val', 'test']
# metrics = ['PLCC', 'RMSE']

# markers_on = {losses[0]: [[[9], [9]], [[11], [11]], [[11], [11]]],
#               losses[1]: [[[8], [8]], [[8], [8]], [[10], [10]]],
#               losses[2]: [[[1], [1]], [[1], [1]], [[1], [1]]]}
# yrange = [(0.7, 1), (0, 15)]


# fig, axs = plt.subplots(3, 2, figsize=(12,12))

# legends = ()
# for j, metric in enumerate(metrics):
# 	for i, stage in enumerate(stages):
# 		for loss in losses:
# 			data = pd.read_csv('csv/loss={}-{}_KonIQ-10k_{}.csv'.format(loss, stage, metric))
# 			epoch = np.asarray(data.iloc[:,1])
# 			performance = np.asarray(data.iloc[:,2])
# 			baseline = np.repeat(performance[0], len(epoch))
# 			l, = axs[i][j].plot(epoch, performance, zorder=1, linewidth=linewidth, color=colors[loss])

# 			if j == 1 and i == 2:
# 				legends = legends + (l, )
# 			if loss == 'norm-in-norm':
# 				b, = axs[i][j].plot(epoch, baseline, '--', color='grey', alpha=0.5, zorder=2, linewidth=linewidth)
# 			axs[i][j].plot(markers_on[loss][i][j][0], performance[markers_on[loss][i][j][0]-1], marker='o', alpha=0.8, zorder=3, markersize=16, color=colors[loss])
# 		axs[i][j].set_xlabel('Epoch', fontsize=fontsize)
# 		axs[i][j].set_ylabel('{} {}'.format(stage,metric), fontsize=fontsize)
# 		axs[i][j].tick_params(axis="x", labelsize=fontsize) 
# 		axs[i][j].tick_params(axis="y", labelsize=fontsize) 
# 		if j == 1:
# 			axs[i][j].set_yticks([0, 5, 10, 15]) 
# 		axs[i][j].set_xlim((0, 30))
# 		axs[i][j].set_ylim(yrange[j])
# 		axs[i][j].grid(True)			
# fig.legend(legends, legend_names, 'lower center', ncol=3, bbox_to_anchor=(0, 1.01, 1, 0.1), fontsize=2+fontsize)
# plt.tight_layout()
# plt.savefig("loss1.pdf", bbox_inches="tight")



# ######
# losses = ['mae', 'mse', 'norm-in-norm']
# legend_names = ('MAE', 'MSE', 'Norm-in-Norm')
# stages = ['train', 'val', 'test']
# metrics = ['SROCC', 'PLCC', 'RMSE']

# markers_on = {losses[0]: [[[10], [9], [9]], [[11], [11], [11]], [[11], [11], [11]]],
#               losses[1]: [[[8], [8], [8]], [[8], [8], [8]], [[10], [10], [10]]],
#               losses[2]: [[[1], [1], [1]], [[1], [1], [1]], [[1], [1], [1]]]}
# yrange = [(0.7, 1), (0.7, 1), (0, 15)]

# fig, axs = plt.subplots(3, 3)

# legends = ()
# for i, metric in enumerate(metrics):
# 	for j, stage in enumerate(stages):
# 		for loss in losses:
# 			data = pd.read_csv('csv/loss={}-{}_KonIQ-10k_{}.csv'.format(loss, stage, metric))
# 			epoch = np.asarray(data.iloc[:,1])
# 			performance = np.asarray(data.iloc[:,2])
# 			baseline = np.repeat(performance[0], len(epoch))
# 			l, = axs[i][j].plot(epoch, performance, zorder=1)
# 			# l, = axs[i][j].plot(epoch, performance, marker='o', markerfacecolor='none', markevery=[markers_on[loss][j][i][0]-1])

# 			if j == 2 and i == 2:
# 				legends = legends + (l, )
# 			if loss == 'norm-in-norm':
# 				b, = axs[i][j].plot(epoch, baseline, '--', color='grey', alpha=0.5, zorder=2)
# 			axs[i][j].plot(markers_on[loss][j][i][0], performance[markers_on[loss][j][i][0]-1], marker='o', markerfacecolor='none', alpha=0.8, zorder=3)
# 		axs[i][j].set_xlabel('Epoch')
# 		axs[i][j].set_ylabel('{} {}'.format(stage,metric))
# 		axs[i][j].set_xlim((0, 30))
# 		axs[i][j].set_ylim(yrange[i])
# 		axs[i][j].grid(True)
				
# fig.legend(legends, legend_names, 'lower center', ncol=3, bbox_to_anchor=(0, 1.01, 1, 0.1)) #
# plt.tight_layout()
# plt.savefig("loss_old.pdf", bbox_inches="tight")
