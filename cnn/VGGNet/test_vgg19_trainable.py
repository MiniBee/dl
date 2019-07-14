#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: utils.py
# @time: 2019/7/4 18:34
# @desc:

import tensorflow as tf
import vgg19_trainable as vgg19
import utils

img1 = utils.load_image('./test_data/tiger.jpeg')
img1_true_result = [1 if i = 292 else 0 for i in range(1000)]

batch1 = img1.reshape((1, 224, 224, 3))

with tf.device('/cpu:0'):
	sess = tf.Session()




