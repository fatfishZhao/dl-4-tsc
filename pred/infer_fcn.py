from classifiers.fcn import Classifier_FCN
import keras
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "4"
set_session(tf.Session(config=config))

dataset = 'NetFlow'
model = keras.models.load_model('/data3/zyx/project/tsc/data/results/fcn/mts_archive_itr_8/'+dataset+'/best_model.hdf5')
x_train = np.load('/data3/zyx/project/tsc/data/archives/mts_archive/'+dataset+'/x_train.npy')
y_train = np.load('/data3/zyx/project/tsc/data/archives/mts_archive/'+dataset+'/y_train.npy')
x_test = np.load('/data3/zyx/project/tsc/data/archives/mts_archive/'+dataset+'/x_test.npy')
y_test = np.load('/data3/zyx/project/tsc/data/archives/mts_archive/'+dataset+'/y_test.npy')
y_test[y_test==13]=2
y_pred = model.predict(x_test)
y_pred = [np.argmax(y_solo)+1 for y_solo in y_pred]
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test,y_pred))
