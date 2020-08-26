import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams['pdf.fonttype'] = 42

image_sizes = ['224x224', '512x384', '1024x768']
legend_names = ('Image size: 224x224', 'Image size: 512x384', 'Image size: 1024x768')

fig, axs = plt.subplots(1, 2)

legends = ()
for image_size in image_sizes:
	data = pd.read_csv('csv/{}-val_KonIQ-10k_SROCC.csv'.format(image_size))
	epoch = np.asarray(data.iloc[:,1])
	SROCC = np.asarray(data.iloc[:,2])

	l1, = axs[0].plot(epoch, SROCC)
	legends = legends + (l1, )

	data = pd.read_csv('csv/{}-val_KonIQ-10k_PLCC.csv'.format(image_size))
	epoch = np.asarray(data.iloc[:,1])
	PLCC = np.asarray(data.iloc[:,2])
	l2, = axs[1].plot(epoch, PLCC)

axs[0].set_xlabel('training epoch')
axs[0].set_ylabel('SROCC')
axs[0].set_xlim((0, 30))
axs[0].set_ylim((0.78, 0.90))
axs[0].grid(True)
axs[1].set_xlabel('training epoch')
axs[1].set_ylabel('PLCC')
axs[1].set_xlim((0, 30))
axs[1].set_ylim((0.83, 0.91))
axs[1].grid(True)
fig.legend(legends, legend_names, 'lower center', ncol=3, bbox_to_anchor=(0, 1.01, 1, 0.1)) #
plt.tight_layout()
plt.savefig("imagesize.pdf", bbox_inches="tight")
