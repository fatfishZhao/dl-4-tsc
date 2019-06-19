import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap


dataset = 'Wafer'
train_x = np.load('/data3/zyx/project/tsc/data/archives/mts_archive/'+dataset+'/x_train.npy')
draw_data = train_x[32]
fig, ax = plt.subplots()
im = ax.imshow(draw_data.T)
cbar = ax.figure.colorbar(im, ax=ax,)
cbar.ax.set_ylabel('value', rotation=-90, va="bottom")
plt.show()