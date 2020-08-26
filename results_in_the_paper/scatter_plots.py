import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['pdf.fonttype'] = 42

fontsize = 12
datasets = ['KonIQ-10k', 'CLIVE']
losses = ['MAE', 'MSE', 'Norm-in-Norm']
colors = {'MAE': 'orange', 'MSE': 'blue', 'Norm-in-Norm': 'green'}
for dataset in datasets:
	for loss in losses:
		data = np.load('npy/{}-{}.npy'.format(dataset, loss), allow_pickle=True)
		x = data.item()['pq']
		y = data.item()['sq']
		s = 16 * np.ones_like(y)
		# plt.scatter(x, y, s=s, alpha=1, c='', edgecolors=colors[loss], marker='.', label=loss)
		plt.scatter(x, y, s=s, color=colors[loss], marker='.', label=loss)
	plt.xlabel("Predicted Quality", fontsize=fontsize)
	plt.ylabel("MOS", fontsize=fontsize)
	plt.legend(loc='upper left', fontsize=fontsize)
	plt.tick_params(axis="x", labelsize=fontsize) 
	plt.tick_params(axis="y", labelsize=fontsize) 
	plt.tight_layout()
	plt.savefig("scatter_{}.pdf".format(dataset), bbox_inches="tight")
	plt.close()