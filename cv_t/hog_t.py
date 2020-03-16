#!/usr/bin/env python
# _*_coding:utf-8 _*_
#@Time    :2019/11/23 下午12:07
#@Author  :hongyue pei 
#@FileName: hog_t.py
#@Software: PyCharm

import pandas as pd
import cv2 as cv
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
from skimage.feature import hog
import functools
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


m_classdict = {'dog': 0, 'cat': 1}

l_samp = os.listdir('../data/dog_cat/train')


pd_sampClass = pd.DataFrame({
    'Sample': list(map(lambda x: os.path.join('../data/dog_cat/train', x), l_samp)),
    'Class': list(map(lambda x: m_classdict[x], list(map(lambda x: x.split('.')[0], l_samp))))
})[['Sample', 'Class']]

pd_sampClass_train, pd_sampClass_cv = train_test_split(pd_sampClass, test_size=0.33, random_state=11)

# fig = plt.figure(figsize=(12, 6))
# for i in range(5):
#     image = cv.imread(pd_sampClass_train['Sample'].iloc[i])
#     image = image[:, :, ::-1]
#     ax = fig.add_subplot(1, 5, i+1)
#     ax.imshow(image)
#     ax.set_title(pd_sampClass_train['Class'].iloc[i])
# plt.show()

fig = plt.figure(figsize=(20, 10))

# img = cv.imread('../data/dog_cat/train/cat.0.jpg')
# img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

l_colorSpace = [cv.COLOR_BGR2GRAY]
l_names = ['GRAY']
l_len = [1]

def get_hog_features(img, orient, pix_per_cell=8, cell_per_block=2, vis=False, feature_vec=True):
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell), cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, visualise=vis, feature_vector=feature_vec)
        return feature_vec, hog_image
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell), cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, visualise=vis, feature_vector=feature_vec)
        return features


def get_features(img, pix_per_cell=8, cell_per_block=2, orient=9, getImage=False, inputFile=True, feature_vec=True):
    l_imgLayers = []
    for cs in l_colorSpace:
        if inputFile:
            l_imgLayers.append(cv.cvtColor(cv.imread(img), cs))
        else:
            l_imgLayers.append(cv.cvtColor(img, cs))
    l_hog_features = []
    l_images = []
    for feature_image in l_imgLayers:
        hog_features = []
        n_channel = 1
        if len(feature_image.shape) > 2:
            n_channel = feature_image.shape[2]
        for channel in range(n_channel):
            featureImg = feature_image
            if n_channel > 2:
                featureImg = feature_image[:, :, channel]
            vout, img = get_hog_features(featureImg, orient, pix_per_cell, cell_per_block, vis=True, feature_vec=feature_vec)
            if getImage:
                l_images.append(img)
            hog_features.append(vout)
        l_hog_features.append(list(hog_features))
    if getImage:
        return l_images
    else:
        return functools.reduce(lambda x, y: x + y, l_hog_features)

if os.path.isfile('../data/dog_cat/x_train.npy') == 0:
    l_x_train = []
    l_x_test = []
    for r in tqdm(pd_sampClass_train.iterrows()):
        l_x_train.append(np.array(get_features(r[1]['Sample'])).ravel())

    for r in tqdm(pd_sampClass_cv.iterrows()):
        l_x_test.append(np.array(get_features(r[1]['Sample'])).ravel())

    x_train = np.array(l_x_train)
    x_test = np.array(l_x_test)
    np.save('../data/dog_cat/x_train.npy', x_train)
    np.save('../data/dog_cat/x_test.npy', x_test)
else:
    x_train = np.load('../data/dog_cat/x_train.npy')
    x_test = np.load('../data/dog_cat/x_test.npy')

y_train = pd_sampClass_train['Class'].values
y_test = pd_sampClass_cv['Class'].values

x_scale = StandardScaler()

x_train = x_scale.fit_transform(x_train)
x_test = x_scale.transform(x_test)

svc = SVC(C=1)
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
print(y_pred)






