#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: aerial_cactus.py
# @time: 2019/7/5 11:34
# @desc:

import cv2 as cv
import pandas as pd
import numpy as np
import os
import tensorflow as tf
import vgg19_trainable as vgg19


class Dataset:
	def __init__(self, x, y):
		self._index_in_epoch = 0
		self._epochs_completed = 0
		self._x = x
		self._y = y
		self._num_examples = x.shape[0]

	def next_batch(self, batch_size, shuffle=True):
		start = self._index_in_epoch
		if start == 0 and self._epochs_completed == 0:
			idx = np.arange(0, self._num_examples)
			np.random.shuffle(idx)
			self._x = self._x[idx]
			self._y = self._y[idx]
		# next batch
		if start + batch_size > self._num_examples:
			self._epochs_completed += 1
			rest_num_examples = self._num_examples - start
			x_rest_part = self._x[start:self._num_examples]
			y_rest_part = self._y[start:self._num_examples]
			idx0 = np.arange(0, self._num_examples)
			np.random.shuffle(idx0)
			start = 0
			self._index_in_epoch = batch_size - rest_num_examples
			x_new_part = self._x[start:self._index_in_epoch]
			y_new_part = self._y[start:self._index_in_epoch]
			return np.concatenate((x_rest_part, x_new_part), axis=0), np.concatenate((y_rest_part, y_new_part), axis=0)
		else:
			self._index_in_epoch += batch_size
			end = self._index_in_epoch
			return self._x[start:end], self._y[start:end]


def load_data(x_path, y):
	file_list = os.listdir(x_path)
	ret = []
	for file in file_list:
		name = file
		file = x_path + file
		ret.append([name, cv.imread(file)])
	data = pd.DataFrame(ret, columns=['id', 'pixel_array'])
	y_data = pd.read_csv(y_file)
	
	data = data.merge(y_data, on='id')
	print(data.head())
	x_list = data[['pixel_array']].values
	y_list = data[['has_cactus']].values
	x_list = np.array([i[0] for i in x_list])
	return x_list, y_list


def train1(data):
	sess = tf.Session()
	images = tf.placeholder(tf.float32, [None, 32, 32, 3])
	true_out = tf.placeholder(tf.float32, [None, 1])
	train_mode = tf.placeholder(tf.bool)
	vgg = vgg19.Vgg19()
	vgg.build(images, train_mode)
	print(vgg.get_var_count())
	sess.run(tf.global_variables_initializer())

	for epoch in range(10):
		print('epoch: {}...'.format(epoch))
		x_list, y_list = data.next_batch(100)
		prob = sess.run(vgg.prob, feed_dict={images: x_list, train_mode: False})
		correct_prediction = tf.equal(tf.argmax(prob, 1), y_list)
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		accuracy = sess.run(accuracy, feed_dict={images: x_list, true_out: y_list, train_mode: False})
		print('accuracy1= ', accuracy)
		
		cost = tf.reduce_sum((vgg.prob - true_out)**2)
		train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
		sess.run(train, feed_dict={images: x_list, true_out: y_list, train_mode: True})
		prob = sess.run(vgg.prob, feed_dict={images: x_list, train_mode: False})
		correct_prediction = tf.equal(tf.argmax(prob, 1), y_list)
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		accuracy = sess.run(accuracy)
		# print(prob)
		print('accuracy2= ', accuracy)


if __name__ == '__main__':
	# file = x_path + '0a1b6731bda8f1b6a807fffc743f8d22.jpg'
	# ret = cv.imread(file)
	x_path = '../../../Aerial_Cactus_train/'
	y_file = '../../../Aerial_Cactus_train.csv'
	x_list, y_list = load_data(x_path, y_file)
	data = Dataset(x_list, y_list)

	train1(data)



	







