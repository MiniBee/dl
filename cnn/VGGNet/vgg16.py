#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: vgg16.py
# @time: 2019/7/1 16:34
# @desc:

import inspect
import os

import numpy as np
import tensorflow as tf
import time

VGG_MEAN = [103.939, 116,779, 123.68]

class Vgg16:
	def __init__(slef, vgg16_npy_path=None):
		if vgg16_npy_path is None:
			path = inspect.getfile(Vgg16)
			path = os.path.abspath(os.path.join(path, os.pardir))
			path = os.path.join(path, 'vgg16.npy')
			vgg16_npy_path = path
			print(path)

		self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
		print('npy file loaded')

	def build(self, rgb):
		'''
		load variable from npy to build the VGG
		param: rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
		'''

		start_time = time.time()
		print('build model started')
		rgb_scaled = rgb * 255.0

		red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
		assert red.get_shape().as_list()[1:] == [224, 224, 1]
		assert green.get_shape().as_list()[1:] == [224, 224, 1]
		assert blue.get_shape().as_list()[1:] == [224, 224, 1]
		bgr = tf.concat(axis=3, values=[blue - VGG_MEAN[0], green - VGG_MEAN[1], red - VGG_MEAN[2]])
		assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

		def conv_layer(self, bottom, name):
			with tf.variable_scope(name) as scope:
				


if __name__ == '__main__':
	Vgg16()








