# FCN
import keras
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Dense, LSTM, multiply, concatenate, Activation, Masking, Reshape
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout

from utils.constants import MAX_NB_VARIABLES, NB_CLASSES_LIST, MAX_TIMESTEPS_LIST, TIME_STEP, LSTM_CELL_NO
from utils.keras_utils import train_model, evaluate_model, set_trainable
from utils.layer_utils import AttentionLSTM

from utils.utils import save_logs

def squeeze_excite_block(input):
    ''' Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor

    Returns: a keras tensor
    '''
    filters = input._keras_shape[-1] # channel_axis = -1 for TF

    se = GlobalAveragePooling1D()(input)
    se = Reshape((1, filters))(se)
    se = Dense(filters // 16,  activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    se = multiply([input, se])
    return se
class Classifier_FCN:

    def __init__(self, output_directory, input_shape, nb_classes, dataset_name,  verbose=False):
        self.output_directory = output_directory
        self.dataset_name = dataset_name
        self.model = self.build_model(input_shape, nb_classes)
        if (verbose == True):
            self.model.summary()
        self.verbose = verbose
        self.model.save_weights(self.output_directory + 'model_init.hdf5')

    def build_model(self, input_shape, nb_classes):
        ip = Input(shape=(input_shape[0], input_shape[1]))

        ''' sabsample timesteps to prevent OOM due to Attention LSTM '''
        # stride = 3
        #
        # x = Permute((2, 1))(ip)
        # x = Conv1D(input_shape[0] // stride, 8, strides=stride, padding='same', activation='relu', use_bias=False,
        #            kernel_initializer='he_uniform')(x)  # (None, variables / stride, timesteps)

        x = Masking()(ip)
        x = AttentionLSTM(LSTM_CELL_NO[self.dataset_name], unroll=True)(x)
        x = Dropout(0.8)(x)

        y = Permute((2, 1))(ip)
        y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = squeeze_excite_block(y)

        y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = squeeze_excite_block(y)

        y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)

        y = GlobalAveragePooling1D()(y)

        x = concatenate([x, y])

        out = Dense(nb_classes, activation='softmax')(x)

        model = Model(ip, out)
        model.summary()

        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=1e-3),
                      metrics=['accuracy'])


        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=100,
                                      factor=1./np.cbrt(2), cooldown=0, min_lr=1e-4)

        file_path = self.output_directory + 'best_model.hdf5'

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path,
                                           monitor='loss', save_best_only=True)

        self.callbacks = [reduce_lr, model_checkpoint]


        return model

    def fit(self, x_train, y_train, x_test, y_test, y_true):
        # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=0,
        #                                                   stratify=y_train)
        x_val = x_test; y_val = y_test
        # x_val and y_val are only used to monitor the test loss and NOT for training
        batch_size = 128
        nb_epochs = 1000

        # mini_batch_size = int(min(x_train.shape[0] / 10, batch_size))
        mini_batch_size = batch_size

        start_time = time.time()

        hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
                              verbose=self.verbose, validation_data=(x_val, y_val), callbacks=self.callbacks)

        duration = time.time() - start_time

        # model = keras.models.load_model(self.output_directory + 'best_model.hdf5')
        self.model.load_weights(self.output_directory + 'best_model.hdf5')

        y_pred = self.model.predict(x_test)

        # convert the predicted from binary to integer
        y_pred = np.argmax(y_pred, axis=1)

        save_logs(self.output_directory, hist, y_pred, y_true, duration)

        keras.backend.clear_session()


