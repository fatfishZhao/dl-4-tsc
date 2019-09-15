import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
import cv2


dataset = 'KickvsPunch'
train_x = np.load('/data3/zyx/project/tsc/data/archives/mts_archive/'+dataset+'/x_train.npy')
draw_data = train_x[8]
draw_data = cv2.resize(draw_data, (224,224), interpolation=cv2.INTER_CUBIC)
c = plt.cm.Blues
fig, ax = plt.subplots()
im = ax.imshow(draw_data.T, cmap=c)
cbar = ax.figure.colorbar(im, ax=ax,)
cbar.ax.set_ylabel('value', rotation=-90, va="bottom")
plt.show()




ax = plt.subplot()
ax.matshow(draw_data.T, cmap=c)
plt.show()
