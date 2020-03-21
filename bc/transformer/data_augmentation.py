#!/usr/bin/env python
# _*_coding:utf-8 _*_
#@Time    :2020/3/21 下午9:02
#@Author  :hongyue pei 
#@FileName: data_augmentation.py
#@Software: PyCharm

from gensim import corpora
from gensim.models import word2vec



def _word2vec(tokens_cut):
    model = word2vec.Word2Vec(tokens_cut, min_count=1)


