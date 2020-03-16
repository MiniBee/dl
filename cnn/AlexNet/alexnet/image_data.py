#!/usr/bin/env python
# _*_coding:utf-8 _*_
#@Time    :2019/10/20 上午11:16
#@Author  :hongyue pei 
#@FileName: image_data.py
#@Software: PyCharm

import cv2 as cv
import random
import numpy as np
import os
import logging

class Image_Data(object):
    def __init__(self, path, batch_size):
        self.batch_size = batch_size
        self.image_size = (227, 227, 3)
        self.base_path = path
        self.test_path = os.path.join(path, 'test')
        self.train_path = os.path.join(path, 'train')

    def get_label(self, image):
        pass

if __name__ == '__main__':
    image_data = Image_Data('/home/peihongyue/project/python/dl/data/dog_cat/dog_cat', 32)
    train_list = os.listdir(image_data.train_path)
    print(train_list)
    for image in train_list[:10]:
        i = cv.imread(os.path.join(image_data.train_path, image))
        print(i)








