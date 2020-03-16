#!/usr/bin/env python
# _*_coding:utf-8 _*_
#@Time    :2020/3/2 上午10:27
#@Author  :hongyue pei 
#@FileName: cifar10_1.py
#@Software: PyCharm

import tensorflow as tf


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train / 255.0
x_test = x_test/ 255.0

fnn = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(32, 32, 3)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

fnn.summary()
fnn.compile(optimizer='adam', loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
fnn.fit(x_train, y_train, banch_size=100, epochs=1, validation_data=(x_test, y_test))

tf.keras.layers.Conv3D(64, (3, 3), activation='relu')

