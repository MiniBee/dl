#!/usr/bin/env python
# _*_coding:utf-8 _*_
#@Time    :2019/12/5 下午9:27
#@Author  :hongyue pei 
#@FileName: cifar_.py
#@Software: PyCharm


import numpy as np
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Conv2D, Dropout, Activation, Flatten, MaxPool2D
from keras.utils import np_utils
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

set_session(tf.Session(config=config))
np.random.seed(11)
import random

batch_size = 32
nb_classes = 10
nb_eopch = 50
img_rows, img_cols = 32, 32
nb_filters = [32, 32, 64, 64]
pool_size = (2, 2)
kernel_size = (3, 3)

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

imput_shape = (img_rows, img_cols, 3)
print(imput_shape)

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

datagen = ImageDataGenerator()
datagen.fit(X_train)

model = Sequential()
model.add(Conv2D(nb_filters[0], kernel_size, padding='same', input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(nb_filters[1], kernel_size))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Conv2D(nb_filters[2], kernel_size, padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(nb_filters[3], kernel_size))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dropout(nb_classes))
model.add(Activation('softmax'))

adam = Adam(lr=0.0001)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

best_model = ModelCheckpoint('cifar10_best.h5', monitor='val_loss', verbose=0, save_best_only=True)
tb = TensorBoard(log_dir='./logs')
model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                    steps_per_epoch=X_train.shape[0] // batch_size,
                    epochs=nb_eopch, verbose=1, validation_data=(X_test, Y_test),
                    callbacks=[best_model, tb])

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score: ', score[0])
print('Accuracy: %.2f%' % (score[1] * 100))
print('Compiled!')










