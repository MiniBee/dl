#!/usr/bin/env python
# _*_coding:utf-8 _*_
#@Time    :2019/12/7 下午12:29
#@Author  :hongyue pei 
#@FileName: cat_dog.py
#@Software: PyCharm


import os
import cv2
import random
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, Flatten, MaxPool2D, Input
from keras.models import Model

import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定GPU
config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
config.gpu_options.allow_growth = True  # 按需

set_session(tf.Session(config=config))

path = '../data/dog_cat/train/'
# filenames = os.listdir(path)
# plt.figure(figsize=(12, 10))
# for i, filename in enumerate(random.sample(filenames, 12)):
#     plt.subplot(3, 4, i + 1)
#     plt.imshow(cv2.imread(os.path.join(path, filename))[:, :, ::-1])
#
# plt.show()

n = 25000
width = 128

X = np.zeros((n, width, width, 3), dtype=np.uint8)
y = np.zeros((n, ), dtype=np.uint8)

for i in tqdm.tqdm(range(int(n/2))):
    X[i] = cv2.resize(cv2.imread(path + 'cat.%d.jpg' % i), (width, width))
    X[i+int(n/2)] = cv2.resize(cv2.imread(path + 'dog.%d.jpg' % i), (width, width))

y[n//2:] = 1

# ---test ---
# index = random.sample(range(n), 200)
# # print(index)
# X = X[index]
# y = y[index]
# ---test ---

# plt.figure(figsize=(12, 10))
# for i in range(12):
#     random_index = random.randint(0, n-1)
#     plt.subplot(3, 4, i+1)
#     plt.imshow(X[random_index][:, :, ::-1])
#     plt.title(['cat', 'dog'][y[random_index]])
#
# plt.show()

from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

input_shape = (128, 128, 3)
img_input = Input(shape=input_shape)
x = img_input
x = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', name='block1_conv1')(x)
x = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', name='block1_conv2')(x)
x = BatchNormalization()(x)
x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='block1_pool')(x)

x = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', name='block2_conv1')(x)
x = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', name='block2_conv2')(x)
x = BatchNormalization()(x)
x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='block2_pool')(x)

x = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', name='block3_conv1')(x)
x = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', name='block3_conv2')(x)
x = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', name='block3_conv3')(x)
x = BatchNormalization()(x)
x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='block3_pool')(x)

x = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', name='block4_conv1')(x)
x = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', name='block4_conv2')(x)
x = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', name='block4_conv3')(x)
x = BatchNormalization()(x)
x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='block4_pool')(x)

x = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', name='block5_conv1')(x)
x = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', name='block5_conv2')(x)
x = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', name='block5_conv3')(x)
x = BatchNormalization()(x)
x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='block5_pool')(x)

x = Flatten(name='flatten')(x)
x = Dense(4096, activation='relu', name='fc1')(x)
x = BatchNormalization()(x)
x = Dense(4096, activation='relu', name='fc2')(x)
x = Dense(1, activation='sigmoid', name='predication')(x)

model = Model(img_input, x, name='vgg16')
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())
# from IPython.display import SVG
# from keras.utils.vis_utils import model_to_dot
#
# SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))


h = model.fit(X_train, y_train, batch_size=32, epochs=20, validation_data=(X_valid, y_valid))

plt.figure(figsize=(10, 4))
plt.subplot(1,2,1)
plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')

plt.subplot(1,2,2)
plt.plot(h.history['accuracy'])
plt.plot(h.history['val_accuracy'])
plt.legend(['acc', 'val_acc'])
plt.ylabel('acc')
plt.xlabel('epoch')

plt.show()









