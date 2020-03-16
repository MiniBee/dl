#!/usr/bin/env python
# _*_coding:utf-8 _*_
#@Time    :2020/3/5 下午4:14
#@Author  :hongyue pei 
#@FileName: similarity.py
#@Software: PyCharm

import numpy as np


def weighted(similarityVec, start, end, i):
    ret = 0
    for j in range(start, end):
        w = 1.0 # / (abs(i - j) + 1)
        ret += w * similarityVec[j]
    return ret / (end - start - 1)

def smooth(similarityVec, k=5):
    buffer = k // 2
    ret = []
    for i in range(len(similarityVec)):
        if i < buffer and i + buffer < len(similarityVec):
            ret.append(weighted(similarityVec, 0, i+buffer + 1, i))
        if i >= buffer and i + buffer < len(similarityVec):
            ret.append(weighted(similarityVec, i-buffer, i+buffer + 1, i))
        if i < buffer and i + buffer >= len(similarityVec):
            ret.append(weighted(similarityVec, 0, len(similarityVec), i))
        if i >= buffer and i + buffer >= len(similarityVec):
            ret.append(weighted(similarityVec, i-buffer, len(similarityVec), i))
    return ret


def euclideanDistance(v1, v2):
    d = 0
    for a, b in zip(v1, v2):
        d += (a - b) ** 2
    return d ** 0.5


def cosineDistance(v1, v2):
    sum12 = 0
    norm1 = 0
    norm2 = 0
    for a, b in zip(v1, v2):
        sum12 += a*b
        norm1 += a ** 2
        norm2 += b ** 2
    if norm1 == 0 or norm2 == 0:
        return None
    return sum12 / ((norm1 * norm2) ** 0.5)


def pearsonDistance(v1, v2):
    return 0.5 + 0.5 * np.corrcoef(v1, v2, rowvar=0)[0][1]


if __name__ == '__main__':
    similarityVec = [0.0022994475929417725, 0.0032742077743830667, 0.011186184419307596, 0.0191241022797853, 0.019969805680894834, 0.0152515180210362, 0.015099219722279256, 0.02104670266894603, 0.028801010059931218, 0.01473930705421799, 0.014486450525549362, 0.02659213840248395, 0.019185621957106862]
    smooth(similarityVec)


