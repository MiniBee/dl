#!/usr/bin/env python
# _*_coding:utf-8 _*_
#@Time    :2019/7/14 下午9:20
#@Author  :hongyue pei 
#@FileName: digitRecognize.py
#@Software: PyCharm

import tensorflow as tf
import pandas as pd
import numpy as np
import vgg16

def load_data(path=''):
    data = pd.read_csv(path)
    return data

def reshape(data_x, height, width):
    return data_x.reshape(data_x.shape[0], height, width, 1)

def get_train(path):
    train_data = load_data(path)
    train_data_x = train_data.as_matrix(columns=train_data.columns[1:])
    train_data_y = train_data.as_matrix(columns=train_data.columns[:1])
    train_data_y = np.eye(10)[train_data_y]
    return reshape(train_data_x, 28, 28), train_data_y

def get_test(path):
    train_data = load_data(path)
    train_data_x = train_data.as_matrix(columns=train_data.columns)
    return reshape(train_data_x, 28, 28)


class Dataset(object):
    def __init__(self, x, y):
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._num_examples = x.shape[0]
        self.x = x
        self.y = y

    def next_batch(self, batch_size, shuffle=True):
        start = self._index_in_epoch
        if start == 0 and self._index_in_epoch == 0:
            idx = np.arange(0, self._num_examples)
            np.random.shuffle(idx)
            self.x = self.x[idx]
            self.y = self.y[idx]
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


if __name__ == '__main__':
    train_model = False
    train_path = '/home/peihongyue/project/python/dl/data/digit_recognizer/train_test.csv'
    train_path = '/home/peihongyue/project/python/dl/data/digit_recognizer/train.csv'
    train_x, train_y = get_train(train_path)
    train_x = train_x / 255
    train_data = Dataset(train_x, train_y)
    model = vgg16.Model()
    best_accuracy = 0.0
    if train_model:
        for i in range(1000):
            pic_x, pic_y = train_data.next_batch(1000)
            pic_y = pic_y.reshape(pic_y.shape[0], 10)
            # training ...
            model.sess.run(model.step, feed_dict={model.inputs: pic_x, model.target_onehot: pic_y})
            accuracy = model.sess.run(model.accuracy, feed_dict={model.inputs: pic_x, model.target_onehot: pic_y})
            loss = model.sess.run(model.loss, feed_dict={model.inputs: pic_x, model.target_onehot: pic_y})
            if i % 10 == 0:
                print('accuracy' + str(i) + ': ', accuracy)
                print('loss' + str(i) + ': ', )
            if accuracy - best_accuracy > 0.03:
                best_accuracy = accuracy
                model.saver.save(model.sess, './model/my-model', global_step=111)
        print(best_accuracy)
    else:
        test_path = '/home/peihongyue/project/python/dl/data/digit_recognizer/test.csv'
        # 28000
        test_x = get_test(test_path)
        test_y_pred = []
        model.init_sess('./model/my-model-111')
        # 每次预测100个
        start = 0
        for i in range(280):
            end = (i + 1) * 100
            print(start, end)
            pic_x = test_x[start:end]
            start = end
            print(pic_x.shape)
            pic_y = model.sess.run(model.y_pred, feed_dict={model.inputs: pic_x})
            pic_y = [list(i).index(max(i)) for i in pic_y]
            test_y_pred.extend(list(pic_y))
        print(test_y_pred)
        data = pd.DataFrame()
        data['ImageId'] = list(range(1, 28001))
        data['Label'] = test_y_pred
        print(data.head())
        data.to_csv('ret.csv', index=False)




