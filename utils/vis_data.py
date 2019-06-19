import numpy as np
import matplotlib.pyplot as plt
import os

dataset = 'KickvsPunch'
train_x = np.load('/data3/zyx/project/tsc/data/archives/mts_archive/'+dataset+'/x_train.npy')
train_y = np.load('/data3/zyx/project/tsc/data/archives/mts_archive/'+dataset+'/y_train.npy')
unique_y = np.unique(train_y)
if not os.path.exists('./'+dataset+ '/'+str(unique_y[0])):
    for each_y in unique_y:
        os.makedirs('./' + dataset + '/'+str(each_y))
for index, (solo_x, solo_y) in enumerate(zip(train_x, train_y)):
    print(index)
    plt.figure(index)
    # plt.ylim((-20,20))
    plt.plot(solo_x)
    plt.title(str(solo_y))
    plt.grid(ls='--')
    plt.savefig('./'+dataset+'/'+str(solo_y)+'/'+str(index)+'.jpg')
    # plt.show()

print(train_x)