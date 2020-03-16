#!/usr/bin/env python
# _*_coding:utf-8 _*_
#@Time    :2020/2/26 下午7:44
#@Author  :hongyue pei 
#@FileName: sif.py
#@Software: PyCharm

import numpy as np
from sklearn.decomposition import TruncatedSVD


class SIF(object):
    def __init__(self, wordWeight, sentencesVec):
        self.wordWeight = wordWeight
        self.sentencesIdx = sentencesVec
        self.wordVecLength = len(self.sentencesIdx[0][0]) - 1

    def sentenceVec(self, sentence):
        ret = np.zeros(shape=self.wordVecLength).astype('float32')
        # print(sentence)
        for word in sentence:
            key = word[0]
            vec = word[1:]
            vec = [float(i) for i in vec]
            # print('-----')
            # print(key, vec)
            # print(np.dot(vec, self.wordWeight.get(key) or 0))
            ret = np.add(ret, np.dot(vec, self.wordWeight.get(key) or 0))
        # print('---', len(sentence))
        # print(ret)
        return ret / len(sentence)

    def getSentencesVec(self):
        ret = []
        for sentence in self.sentencesIdx:
            ret.append(self.sentenceVec(sentence))
        # print(np.array(ret).shape)
        return np.array(ret)

    def getPC(self, sentencesVec):
        svd = TruncatedSVD(n_components=1)
        # print(sentencesVec)
        svd.fit(sentencesVec)
        return svd.components_

    def removePC(self, sentencesVec):
        pc = self.getPC(sentencesVec)
        sentencesVec = sentencesVec - np.dot(sentencesVec, np.dot(pc.T, pc))
        return sentencesVec

    def __call__(self, *args, **kwargs):
        sentencesVec = self.getSentencesVec()
        ret = self.removePC(sentencesVec)
        return ret

if __name__ == '__main__':
    pass


