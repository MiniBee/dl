#!/usr/bin/env python
# _*_coding:utf-8 _*_
#@Time    :2019/12/3 下午9:57
#@Author  :hongyue pei 
#@FileName: tf_test6.py
#@Software: PyCharm

import tensorflow as tf
import cv2 as cv
from sklearn.utils import shuffle
import numpy as np


def get_image(path, shape=None):
    image = cv.imread(path)
    image = image[:, :, ::-1]
    if shape != None:
        image = cv.resize(image, shape)
    return image


def image_generator(pd_input, shape=(64, 64), batch_size = 32):
    num_samples = pd_input.shape[0]
    while 1:
        pd_input_shuffle = shuffle(pd_input)
        for offset in range(0, num_samples, batch_size):
            l_x = []
            l_y = []
            batch_samples = pd_input_shuffle.iloc[offset:offset + batch_size]
            for idx in range(batch_size):
                try:
                    path = batch_samples.iloc[idx]['Sample']
                    label = batch_samples.iloc[idx]['Class']
                    image = get_image(path, shape)
                    l_x.append(image)
                    l_y.append(label)
                except:
                    pass
            np_x = np.array(l_x)
            np_y = np.array(l_y)
            yield shuffle(np_x, np_y)


if __name__ == '__main__':
    l_imagepath = []
    l_img = []
    for image_path in l_imagepath:
        img = get_image(image_path)
        l_img.append(img)



