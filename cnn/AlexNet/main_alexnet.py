#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: main_alexnet.py
# @time: 2019/6/25 22:16
# @desc:

import os
import numpy as np
import tensorflow as tf
from alexnet import alexnet
from datagenerator import ImageDataGenerator
from datetime import datetime
import glob
from tensorflow.data import Iterator


def main():
	learning_rate = 1e-3
	num_epochs = 10
	train_batch_size = 100
	test_batch_size = 10
	dropout_rate = 0.5
	num_classes = 2
	display_step = 2

	filewriter_path = './model/tensorboard/'
	checkpoint_path = './model/checkpoints/'

	image_format = 'jpg'
	file_name_of_class = ['cat', 'dog']
	train_dataset_paths = ['./data/train/cat/', './data/train/dog/']
	test_dataset_paths = ['./data/test/cat/', './data/test/dog/']

	train_image_pathes = []
	train_labels = []

	for train_dataset_path in train_dataset_paths:
		length = len(train_image_pathes)
		train_image_pathes[length:length] = np.array(glob.glob(train_dataset_path + '*.' + image_format)).tolist()
	for image_path in train_image_pathes:
		image_file_name = image_path.split('/')[-1]
		for i in range(num_classes):
			if file_name_of_class[i] in image_file_name:
				train_labels.append(i)
				break

	test_image_paths = []
	test_labels = []
	for test_dataset_path in test_dataset_paths:
		length = len(test_image_paths)
		test_image_paths[length:length] = np.array(glob.glob(test_dataset_path + '*.' + image_format)).tolist()
	for image_path in test_image_paths:
		iamge_file_name = image_path.split('/')[-1]
		for i in range(num_classes):
			if file_name_of_class[i] in image_file_name:
				test_labels.append(i)
				break

	# get Datasets
	train_data = ImageDataGenerator(
		images = train_image_pathes,
		labels=train_labels,
		batch_size=train_batch_size,
		num_classes=num_classes,
		image_format=image_format,
		shuffle=True)

	test_data = ImageDataGenerator(
		images=test_image_paths,
		labels=test_labels,
		batch_size=test_batch_size,
		num_classes=num_classes,
		image_format=image_format,
		shuffle=False)

	# get Iterators
	with tf.name_scope('input') as scope:
		train_iterator = Iterator.from_structure(train_data.data.output_types, train_data.data.output_shapes)
		training_initalizer = train_iterator.make_initializer(train_data.data)
		test_iterator = Iterator.from_structure(test_data.data.output_types, test_data.data.output_shapes)
		testing_initalizer=test_iterator.make_initializer(test_data.data)

		train_next_batch = train_iterator.get_next()
		test_next_batch = test_iterator.get_next()

	x = tf.placeholder(tf.float32, [None, 227, 227, 3])
	y = tf.placeholder(tf.float32, [None, num_classes])
	keep_prob = tf.placeholder(tf.float32)

	# alexnet
	fc8 = alexnet(x, keep_prob, num_classes)

	# loss
	with tf.name_scope('loss') as scope:
		loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=fc8, labels=y))

	# optimizer
	with tf.name_scope('optimizer') as scope:
		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
		train_op = optimizer.minimize(loss_op)

	# accuracy
	with tf.name_scope('accuracy') as scope:
		correct_pred = tf.equal(tf.argmax(fc8, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	init = tf.global_variables_initializer()
	tf.summary.scalar('loss', loss_op)
	tf.summary.scalar('accuracy', accuracy)
	merged_summary = tf.summary.merge_all()
	writer = tf.summary.FileWriter(filewriter_path)

	saver = tf.train.Saver()

	train_batches_per_epoch = int(np.floor(train_data.data_size / train_batch_size))
	test_batches_per_epoch = int(np.floor(test_data.data_size / test_batch_size))

	with tf.Session() as sess:
		sess.run(init)
		writer.add_graph(sess.graph)
		print('{}: Start training ...'.format(datetime.now()))
		print('{}. Open Tensorboard at --logdir {}'.format(datetime.now(), filewriter_path))
		for epoch in range(num_epochs):
			sess.run(training_initalizer)
			print('{}: Epoch number: {} start'.format(datetime.now(), epoch + 1))

			for step in range(train_batches_per_epoch):
				img_batch, label_batch = sess.run(train_next_batch)
				print('==========================>', img_batch.shape)
				print('==========================>', label_batch.shape)
				loss, _ = sess.run([loss_op, train_op], feed_dict={x: img_batch, y: label_batch, keep_prob: dropout_rate})
				if step % display_step == 0:
					print('{}: loss = {}'.format(datetime.now(), loss))
					s = sess.run(merged_summary, feed_dict={x: img_batch, y: label_batch, keep_prob: 1})
					writer.add_summary(s, epoch * train_batches_per_epoch + step)

			# accuracy
			print('{}: Start validation'.format(datetime.now()))
			sess.run(testing_initalizer)
			test_acc = 0
			test_count = 0
			for _ in range(test_batches_per_epoch):
				img_batch, label_batch = sess.run(test_next_batch)
				acc = sess.run(accuracy, feed_dict={x: img_batch, y:label_batch, keep_prob: 1})
				test_acc += acc
				test_count += 1
			try:
				test_acc /= test_count
			except:
				print('ZeroDivisionError!')
			print('{}: Validation accuracy = {:.4f}'.format(datetime.now(), test_acc))

			# save model
			print('{}: Saving checkpoints of model ... '.format(datetime.now()))
			checkpoint_name = os.path.join(checkpoint_path, 'model_epoch' + str(epoch + 1) + '.ckpt')
			save_path = saver.save(sess, checkpoint_name)

			# this epoch is over
			print('{}: Epoch number: {} end'.format(datetime.now(), epoch + 1))


if __name__ == '__main__':
	main()








