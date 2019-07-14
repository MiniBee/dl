#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: facialExpression.py
# @time: 2019/7/6 9:18
# @desc:

import cv2 as cv
import pandas as pd
import numpy as np
import vgg16


emotion_norm = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise',
                6: 'Neutral'}

class Dataset:
    def __init__(self, x, y):
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self.x = x
        self.y = y
        self._num_examples = x.shape[0]

    def next_batch(self, batch_size, shuffle=True):
        start = self._index_in_epoch
        if start == 0 and self._epochs_completed == 0:
            idx = np.arange(0, self._num_examples)
            np.random.shuffle(idx)
            self.x = self.x[idx]
            self.y = self.y[idx]
        # next batch
        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start
            x_rest_part = self.x[start:self._num_examples]
            y_rest_part = self.y[start:self._num_examples]
            idx0 = np.arange(0, self._num_examples)
            np.random.shuffle(idx0)
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            x_new_part = self.x[start:self._index_in_epoch]
            y_new_part = self.y[start:self._index_in_epoch]
            return np.concatenate((x_rest_part, x_new_part), axis=0), np.concatenate((y_rest_part, y_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self.x[start:end], self.y[start:end]


def load_data(path='/data/phy/AIE16/6.22/01/data/test.csv'):
    '''
    :param path: data path
    :return: ['emotion', 'pixels', 'Usage', 'label', 'true_out']
    '''
    data = pd.read_csv(path)
    data['label'] = data['emotion'].apply(lambda x: emotion_norm[x])
    label = pd.get_dummies(data[['label']])
    # label one hot
    label['true_out'] = label.apply(lambda x: [x['label_Angry'], x['label_Disgust'],
                                               x['label_Fear'], x['label_Happy'],
       x['label_Neutral'], x['label_Sad'], x['label_Surprise']], axis=1)
    data = pd.concat([data, label[['true_out']]], axis=1)
    # pixels reshape
    data['pixels'] = data['pixels'].apply(lambda x: np.reshape([int(i) for i in x.split(' ')], (48, 48)))
    return data


if __name__ == '__main__':
    print('hello ... ')
    data = load_data('../../data/fer2013/fer2013.csv')
    print('split test train data ... ')
    data['pixels'] = data['pixels'] / 255
    test_data = data[['true_out', 'pixels']][:2000]
    train_data = data[['true_out', 'pixels']][2000:]
    pic = train_data['pixels'].values
    true_out = train_data['true_out'].values
    data = Dataset(pic, true_out)
    test = Dataset(test_data['pixels'].values, test_data['true_out'].values)
    test_pic, test_out = test.next_batch(100)
    test_pic = np.array([np.reshape([x], [48, 48, 1]) for x in test_pic])
    test_out = np.array([x for x in test_out])
    print(data._num_examples)
    model = vgg16.Model()
    best_accuracy = 0.0
    for epoch in range(20000):
        print('epoch ... {}'.format(epoch))
        train_pic, train_to = data.next_batch(64, True)
        train_pic = np.array([np.reshape([x], [48, 48, 1]) for x in train_pic])
        train_to = np.array([x for x in train_to])
        print(train_pic.shape)
        print(train_to.shape)
        model.sess.run(model.step, feed_dict={model.inputs: train_pic, model.target_onehot: train_to})
        # print(model.sess.run(model.inputs, feed_dict={model.inputs: train_pic}))
        # print(model.sess.run(model.target_onehot, feed_dict={model.target_onehot: train_to}))
        # print(model.sess.run(model.y_pred, feed_dict={model.inputs: train_pic, model.target_onehot: train_to}))
        # print(model.sess.run(model.loss, feed_dict={model.inputs: train_pic, model.target_onehot: train_to}))
        accuracy = model.sess.run(model.accuracy, feed_dict={model.inputs: test_pic, model.target_onehot: test_out})
        print('accuracy: ', accuracy)
        if accuracy - best_accuracy > 0.03:
            best_accuracy = accuracy
            model.saver.save(model.sess, './model/my-model', global_step=111)
    print(best_accuracy)






