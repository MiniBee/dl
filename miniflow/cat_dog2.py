#!/usr/bin/env python
# _*_coding:utf-8 _*_
#@Time    :2019/12/7 下午3:26
#@Author  :hongyue pei 
#@FileName: cat_dog2.py
#@Software: PyCharm

from keras.applications import VGG16
from keras.layers import Input, Lambda, GlobalAveragePooling2D, Dense, Dropout
from keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import cv2


path = '../data/dog_cat/train/'
def preprocess_input(x):
    return x - [103.939, 116.779, 123.68]

# plt.figure(figsize=(12, 10))
# for i in range(12):
#     random_index = random.randint(0, n-1)
#     plt.subplot(3, 4, i+1)
#     plt.imshow(X[random_index][:, :, ::-1])
#     plt.title(['cat', 'dog'][y[random_index]])
#
# plt.show()
width = 224

cnn_model = VGG16(include_top=False, input_shape=(width, width, 3), weights='imagenet')
for layer in cnn_model.layers:
    layer.trainable = False

inputs = Input((width, width, 3))
x = inputs
x = Lambda(preprocess_input, name='preprocessing')(x)
x = cnn_model(x)
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(1, activation='sigmoid')(x)

model = Model(inputs, x)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))

n = 25000

X = np.zeros((n, width, width, 3), dtype=np.uint8)
y = np.zeros((n, ), dtype=np.uint8)

for i in tqdm.tqdm(range(int(n/2))):
    X[i] = cv2.resize(cv2.imread(path + 'cat.%d.jpg' % i), (width, width))
    X[i+int(n/2)] = cv2.resize(cv2.imread(path + 'dog.%d.jpg' % i), (width, width))

y[n//2:] = 1

from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)


h = model.fit(X_train, y_train, batch_size=128, epochs=20, validation_data=(X_valid, y_valid))
plt.figure(figsize=(10, 4))
plt.subplot(1,2,1)
plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')

plt.subplot(1,2,2)
plt.plot(h.history['acc'])
plt.plot(h.history['val_acc'])
plt.legend(['acc', 'val_acc'])
plt.ylabel('acc')
plt.xlabel('epoch')

plt.show()



