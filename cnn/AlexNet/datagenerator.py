#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: alexnet.py
# @time: 2019/6/25 22:18
# @desc:


import tensorflow as tf
import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor
from tensorflow.data import Dataset

VGG_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)


class ImageDataGenerator(object):
	def __init__(self, images, labels, batch_size, num_classes, image_format='jpg', shuffle=True):
		self.img_paths = images
		self.labels = labels
		self.data_size = len(self.labels)
		self.num_classes = num_classes
		self.image_format = image_format
		if shuffle:
			self._shuffle_lists()
		self.img_paths = convert_to_tensor(self.img_paths, dtype=dtypes.string)
		self.labels = convert_to_tensor(self.labels, dtype=dtypes.int32)
		data = tf.data.Dataset.from_tensor_slices((self.img_paths, self.labels))
		data = data.map(self._parse_function_train)
		data = data.batch(batch_size)
		self.data = data

	def _shuffle_lists(self):
		path = self.img_paths
		labels = self.labels
		permutation = np.random.permutation(self.data_size)
		self.img_paths = []
		self.labels = []
		for i in permutation:
			self.img_paths.append(path[i])
			self.labels.append(labels[i])

	def _parse_function_train(self, filename, label):
		one_hot = tf.one_hot(label, self.num_classes)
		img_string = tf.read_file(filename)
		if self.image_format == 'jpg':
			img_decoded = tf.image.decode_jpeg(img_string, channels=3)
		elif self.image_format == 'png':
			img_decoded = tf.image.decode_png(img_string, channels=3)
		else:
			print('Error! Can not confirm the format of images!' )
		img_resized = tf.image.resize_images(img_decoded, [227, 227])
		img_centerd = tf.subtract(img_resized, VGG_MEAN)
		img_bgr = img_centerd[:, :, ::-1]
		return img_bgr, one_hot





