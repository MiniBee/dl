#!/usr/bin/env python
# _*_coding:utf-8 _*_
#@Time    :2019/12/8 下午9:47
#@Author  :hongyue pei 
#@FileName: captcha_.py
#@Software: PyCharm


from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
import numpy as np
import random
import string
from keras import backend as K
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Lambda, Reshape, Dense, Dropout, GRU, add, concatenate
from keras.models import Model, Input



characters = string.digits + string.ascii_uppercase
print(characters)

width, height, n_len, n_class = 170, 80, 4, len(characters) + 1

generator = ImageCaptcha(width=width, height=height)
# random_str = ''.join([random.choice(characters) for j in range(4)])
# img = generator.generate_image(random_str)
#
# plt.imshow(img)
# plt.show()


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


rnn_size = 128
input_tensor = Input((width, height, 3))
x = input_tensor
x = Lambda(lambda x: (x-127.5)/127.5)(x)

for i in range(3):
    for j in range(2):
        x = Conv2D(32 * 2 ** i, 3, kernel_initializer='he_uniform')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

conv_shape = x.get_shape().as_list()
rnn_length = conv_shape[1]
rnn_dimen = conv_shape[2] * conv_shape[3]
print(conv_shape, rnn_length, rnn_dimen)

x = Reshape(target_shape=(rnn_length, rnn_dimen))(x)
rnn_length -= 2


x = Dense(rnn_size, kernel_initializer='he_uniform')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.2)(x)

gru_1 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_uniform', name='gru1')(x)
gru_1b = GRU(rnn_size, return_sequences=True, kernel_initializer='he_uniform', name='gru1b')(x)

x = add([gru_1, gru_1b])

gru_2 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_uniform', name='gru2')(x)
gru_2b = GRU(rnn_size, return_sequences=True, kernel_initializer='he_uniform', name='gru2b')(x)

x = concatenate([gru_2, gru_2b])

x = Dropout(0.2)(x)
x = Dense(n_class, activation='softmax')(x)

base_model = Model(inputs=input_tensor, outputs=x)

labels = Input(shape=[n_len], name='the_labels', dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')

loss_out = Lambda(ctc_lambda_func, output_shape=(1, ), name='ctc')([x, label, input_length, label_length])
model = Model(inputs=[input_tensor, labels, input_length, label_length], outputs=[loss_out])
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')

print(model.summary())


def gen(batch_size=128):
    X = np.zeros((batch_size, width, height, 3), dtype=np.uint8)
    y = np.zeros((batch_size, n_len), dtype=np.uint8)
    generator = ImageCaptcha(width=width, height=height)
    while True:
        for i in range(batch_size):
            random_str = ''.join([random.choice(characters) for j in range(n_len)])
            X[i] = np.array(generator.generate_image(random_str)).transpose(1, 0, 2)
            y[i] = [characters.find(x) for x in random_str]
        yield [X, y, np.ones(batch_size)*rnn_length, np.ones(batch_size)*n_len], np.ones(batch_size)






