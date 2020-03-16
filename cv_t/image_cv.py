#!/usr/bin/env python
# _*_coding:utf-8 _*_
#@Time    :2019/11/21 下午10:36
#@Author  :hongyue pei 
#@FileName: image_cv.py
#@Software: PyCharm

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    img = cv.imread('../data/dog_cat/test/1.jpg')
    # print(img.shape)
    # plt.imshow(img)
    #
    # fig = plt.figure(figsize=(10, 6))
    # ax1 = fig.add_subplot(121)
    # ax2 = fig.add_subplot(122)
    #
    # ax1.imshow(img)
    # # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # # ax2.imshow(cv.resize(img, (300, 300)))
    # # img2 = cv.putText(img, 'dog', (10, 50), cv.FONT_HERSHEY_PLAIN, 4, (0, 0, 0), 2, cv.LINE_AA)
    # img2 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # print(img2.shape)
    # ax2.imshow(img2)

    # img = img[50:250, 100:300]
    # print(img[:, :, 0])
    # plt.imshow(img[:, :, 0])
    # plt.show()
    # fig = plt.figure(figsize=(15, 5))
    # ax1 = fig.add_subplot(131)
    # ax2 = fig.add_subplot(132)
    # ax3 = fig.add_subplot(133)
    # ax1.imshow(img[:, :, 0] > 100)
    img1 = ((img[:, :, 0] > 50) + (img[:, :, 0] < 10)).astype(np.uint8)
    # ax2.imshow((img[:, :, 0] > 50) + (img[:, :, 0] < 10))
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    img_e = cv.erode(img1, kernel, iterations=3)
    img_de = cv.dilate(img_e, kernel, iterations=1)
    img_ede = cv.erode(img_de, kernel, iterations=18)

    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    ax1.imshow(img_e)
    ax2.imshow(img_de)
    ax3.imshow(img_ede)
    plt.show()

    # Sobel
    sobelx = cv.Sobel(img_ede*255, cv.CV_64F, 1, 0)
    sobely = cv.Sobel(img_ede*255, cv.CV_64F, 0, 1)
    img_sob = np.sqrt(sobelx**2 + sobely**2).astype(np.uint8)
    plt.imshow(img_sob)
    plt.show()






