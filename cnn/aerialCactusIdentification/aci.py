#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: aci.py
# @time: 2019/7/16 上午10:36
# @desc:

import tensorflow as tf
import pandas as pd
import numpy as np
import cv2 as cv
import os
import conv


def load_pic(path):
    file_list = os.listdir(path)
    pic_list = []
    name_list = []
    for i, file_name in enumerate(file_list):
        pic_path = path + file_name
        pic_list.append(cv.imread(pic_path))
        name_list.append(file_name)
        # if i > 200:
        #     break
    return np.array(pic_list), np.array(name_list)


def get_label(file):
    data = pd.read_csv(file)
    label_dict = {}
    for detail in data.values:
        label_dict[detail[0]] = detail[1]
    return label_dict


class Dataset(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self._index_in_epochs = 0
        self._num_examples = x.shape[0]

    def next_batch(self, batch_size):
        start = self._index_in_epochs
        if start == 0 and self._index_in_epochs == 0:
            idx = np.arange(0, self._num_examples)
            np.random.shuffle(idx)
            self.x = self.x[idx]
            self.y = self.y[idx]
        if start + batch_size > self._num_examples:
            rest_num_examples = self._num_examples - start
            x_rest_part = self.x[start:rest_num_examples]
            y_rest_part = self.y[start:rest_num_examples]
            idx0 = np.arange(0, self._num_examples)
            np.random.shuffle(idx0)
            start = 0
            self._index_in_epochs = batch_size - rest_num_examples
            x_new_part = self.x[start:self._index_in_epochs]
            y_new_part = self.y[start:self._index_in_epochs]
            return np.concatenate((x_rest_part, x_new_part), axis=0), np.concatenate((y_rest_part, y_new_part), axis=0)
        else:
            self._index_in_epochs += batch_size
            end = self._index_in_epochs
            return self.x[start:end], self.y[start:end]


if __name__ == '__main__':
    train_model = False
    train_label_ = '/home/peihongyue/project/python/dl/data/aerial-cactus-identification/train.csv'
    train_pic_ = '/data/phy/datas/aerial-cactus-identification/train/'
    train_pic_ = '/home/peihongyue/project/python/dl/data/aerial-cactus-identification/train/'
    test_pic_ = '/data/phy/datas/aerial-cactus-identification/test/'
    test_pic_ = '/home/peihongyue/project/python/dl/data/aerial-cactus-identification/test/'
    pic_array, name_array = load_pic(train_pic_)
    label_dict = get_label(train_label_)
    label_array = np.array([label_dict[i] for i in name_array])
    print(pic_array.shape)
    print(label_array.shape)
    dataset = Dataset(pic_array, name_array)
    model = conv.Model(32, 32, 2)
    best_accuracy = 0.0
    if train_model:
        for i in range(20000):
            pic_x, pic_y = dataset.next_batch(128)
            pic_y = np.array([label_dict[i] for i in pic_y])
            pic_y = np.array([[1, 0] if i == 1 else [0, 1] for i in pic_y])
            model.sess.run(model.step, feed_dict={model.inputs: pic_x, model.labels: pic_y})
            accuracy = model.sess.run(model.accuracy, feed_dict={model.inputs: pic_x, model.labels: pic_y})
            loss = model.sess.run(model.loss, feed_dict={model.inputs: pic_x, model.labels: pic_y})
            if i % 10 == 0:
                print('accuracy' + str(i) + ': ', accuracy)
                print('loss' + str(i) + ': ', loss)
            if accuracy - best_accuracy > 0:
                best_accuracy = accuracy
                model.saver.save(model.sess, './model/my-model', global_step=111)
        print(best_accuracy)
    else:
        test_x, test_name = load_pic(test_pic_)
        test_y_pred = []
        model.init_sess('./model/my-model-111')
        # 每次预测100个
        start = 0
        for i in range(40):
            end = (i + 1) * 100
            print(start, end)
            pic_x = test_x[start:end]
            start = end
            print(pic_x.shape)
            pic_y = model.sess.run(model.logits, feed_dict={model.inputs: pic_x})
            pic_y = [1 - list(i).index(max(i)) for i in pic_y]
            test_y_pred.extend(list(pic_y))
        print(len(test_y_pred))
        data = pd.DataFrame()
        data['id'] = list(range(1, 4001))
        data['has_cacus'] = test_y_pred
        print(data.head())
        data.to_csv('./ret.csv', index=False)

