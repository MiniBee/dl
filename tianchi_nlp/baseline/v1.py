#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: v1.py
# @time: 2020/7/27 上午11:05
# @desc:

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np

try:
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

p_config = {'batch_size': 128, 'vocab_size': 8000,
            'embedding_dim': 1024, 'epochs': 2, 'units': 256,
            'max_inp': 1200, 'train_data': '/home/peihongyue/data/tianchi_nlp/train_set.csv',
            'test_data': '/home/peihongyue/data/tianchi_nlp/test_a_sample_submit.csv',
            'model_data': '/home/peihongyue/data/tianchi_nlp/model/'}


def load_data(path):
    y_array = []
    x_array = []
    with open(path) as f:
        f.readline()
        for line in f:
            line = line.split('\t')
            y_array.append(int(line[0]))
            x_array.append([int(i) for i in line[1].split(' ')])
    x_array = tf.keras.preprocessing.sequence.pad_sequences(x_array, maxlen=p_config['max_inp'], padding='post')
    return x_array, np.array(y_array)


def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1200,)),
        tf.keras.layers.Embedding(8000, 1024),
        tf.keras.layers.LSTM(256),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(14, activation='softmax')
    ])
    model.summary()
    return model


def train(model, x_train, y_train, x_test, y_test):
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=128, epochs=20, validation_data=(x_test, y_test), callbacks=[callback], class_weight=class_weight)
    return model


def load_test(path):
    x_array = []
    with open(path) as f:
        f.readline()
        for line in f:
            x_array.append([int(i) for i in line.split(' ')])
    x_array = tf.keras.preprocessing.sequence.pad_sequences(x_array, maxlen=p_config['max_inp'], padding='post')
    return x_array


if __name__ == '__main__':
    train_data='/home/peihongyue/data/tianchi_nlp/train_set.csv'
    test_data='/home/peihongyue/data/tianchi_nlp/test_a_sample_submit.csv'
    x_array, y_array = load_data(train_data)
    print(x_array.shape)
    print(y_array.shape)

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3)
    class_weight = {0: 38918, 1: 36945, 2: 31425, 3: 22133, 4: 15016, 5: 12232, 6: 9985, 7: 8841, 8: 7847, 9: 5878,
                    10: 4920, 11: 3131, 12: 1821, 13: 908}
    c_sum = sum(class_weight.values())
    class_weight = {key: (1 / val) * (c_sum) / 2.0 for key, val in class_weight.items()}

    x_train, x_test, y_train, y_test = train_test_split(x_array, y_array, test_size=0.3)
    model = build_model()
    model = train(model, x_train, y_train, x_test, y_test)

    y_pred = model.predict(x_test)
    y_pred = tf.argmax(y_pred, axis=1).numpy()
    print(y_pred)
    print(y_test)

    print(f1_score(y_pred, y_test, average='macro'))

    test_x = load_test(test_data)
    y_pred = model.predict(test_x)
    y_pred = tf.argmax(y_pred, axis=1).numpy()
    with open('/home/peihongyue/data/tianchi_nlp/ans1.csv', 'w') as f:
        f.write('label' + '\n')
        for y in y_pred:
            f.write(str(y) + '\n')







