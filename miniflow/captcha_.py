#!/usr/bin/env python
# _*_coding:utf-8 _*_
#@Time    :2019/12/8 下午9:47
#@Author  :hongyue pei 
#@FileName: captcha_.py
#@Software: PyCharm

from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

import string

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定GPU
config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
config.gpu_options.allow_growth = True  # 按需

set_session(tf.Session(config=config))


characters = string.digits + string.ascii_uppercase
# print(characters)

width, height, n_len, n_class = 170, 80, 4, len(characters)
generator = ImageCaptcha(width=width, height=height)
random_str = ''.join([random.choice(characters) for j in range(4)])
img = generator.generate_image(random_str)


def gen(batch_size=32):
    X = np.zeros((batch_size, height, width, 3), dtype=np.uint8)
    y = [np.zeros((batch_size, n_class), dtype=np.uint8) for i in range(n_len)]
    generator = ImageCaptcha(width=width, height=height)
    while True:
        for i in range(batch_size):
            random_str = ''.join([random.choice(characters) for j in range(4)])
            X[i] = generator.generate_image(random_str)
            for j, ch in enumerate(random_str):
                y[j][i:] = 0
                y[j][i, characters.find(ch)] = 1
        yield X, y


def decode(y, index=0):
    y = np.argmax(np.array(y), axis=2)[:, index]
    return ''.join([characters[x] for x in y])



# X, y = next(gen(2))
# print(X.shape, len(y), y[0].shape)
# plt.imshow(X[0])
# plt.title(decode(y))
# plt.show()

from keras.layers import Dense, Conv2D, Dropout, Activation, Flatten, MaxPool2D, Input
from keras.models import Model

input_tensor = Input(shape=(height, width, 3))
x = input_tensor
for i in range(4):
    x = Conv2D(filters=32*2**i, kernel_size=3, activation='relu')(x)
    x = Conv2D(filters=32 * 2 ** i, kernel_size=3, activation='relu')(x)
    x = MaxPool2D(pool_size=2)(x)

x = Flatten()(x)
x = Dropout(rate=0.25)(x)
x = [Dense(n_class, activation='softmax', name='c%d' % (i+1))(x) for i in range(4)]

model = Model(inputs=input_tensor, outputs=x)
model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

# from keras.utils.vis_utils import plot_model
# plot_model(model, to_file='model.png', show_shapes=True)

x_valid, y_valid = next(gen(300))

h = model.fit_generator(gen(128), steps_per_epoch=400, epochs=2,
                        workers=4, pickle_safe=True,
                        validation_data=gen(128), validation_steps=10)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.ylim(0, 1)

plt.subplot(1, 2, 2)
for i in range(4):
    plt.plot(h.history['val_c%d_accuracy' % (i + 1)])
plt.legend(['val_c%d_acc' % (i + 1) for i in range(4)])
plt.ylabel('acc')
plt.xlabel('epoch')
plt.ylim(0.9, 1)
plt.show()









