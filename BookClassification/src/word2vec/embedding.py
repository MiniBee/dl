#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: embedding.py
# @time: 2020/9/23 下午1:38
# @desc:

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import models
import gensim
import joblib

from __init__ import *
from src.word2vec.autoencoder import AutoEncoder
from src.utils.config import *
from src.utils.tools import create_logger, query_cut
logger = create_logger(root_path + '/logs/embedding.log')


class SingletonMetaClass(type):
    def __init__(self, *args, **kwargs):
        self.__instance=None
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        if self.__instance is None:
            self.__instance = super(SingletonMetaClass, self).__call__(*args, **kwargs)
            return self.__instance
        else:
            return self.__instance


class Embedding(metaclass=SingletonMetaClass):
    def __init__(self):
        self.stopWords = open(root_path + '/data/stopWords_cn.txt', encoding='utf-8').readlines()
        # self.ae = AutoEncoder(max_len, vocab_size, embedding_dim)

    def load_data(self):
        logger.info('load data ... ')
        self.data = pd.concat([
            pd.read_csv(root_path + '/data/train.csv', sep=',', names=['label', 'text']),
            pd.read_csv(root_path + '/data/test.csv', sep=',', names=['label', 'text'])
        ])
        # self.data['text'] = self.data['title'] + self.data['desc']
        self.data['text'] = self.data['text'].apply(query_cut)
        self.data['text'] = self.data['text'].apply(lambda x: ' '.join(x))
        # print(self.data['text'])

    def trainer(self):
        logger.info('train tfidf ... ')
        # count_vect = TfidfVectorizer(stop_words=self.stopWords, ngram_range=(1,2))
        # self.tfidf = count_vect.fit(self.data['text'])

        logger.info('train word2vec ... ')
        # self.data['text'] = self.data['text'].apply(lambda x: x.split(' '))
        # self.w2v = models.Word2Vec(min_count=2, window=5, size=300, sample=6e-5, max_vocab_size=50000)  # size Dimensionality of the word vectors.
        # self.w2v.build_vocab(self.data['text'])
        # self.w2v.train(self.data['text'], total_examples=self.w2v.corpus_count, epochs=15, report_delay=1)

        logger.info('train fasttext ... ')

        self.fast = models.FastText(self.data['text'].apply(lambda x: x.split(' ')), size=300, window=3, min_count=2, max_vocab_size=50000)

        logger.info('train lda ... ')
        # self.id2word = gensim.corpora.Dictionary(self.data.text)
        # corpus = [self.id2word.doc2bow(text) for text in self.data.text]
        # self.LDAmodel = models.LdaMulticore(corpus=corpus, id2word=self.id2word, num_topics=30)

        logger.info('train autoencoder ... ')
        # self.ae.train(self.data)

    def saver(self):
        logger.info('save autoencoder model ... ')
        # self.ae.save()

        logger.info('save tfidf model ... ')
        # joblib.dump(self.tfidf, root_path + '/model/embedding/tfidf')

        logger.info('save w2v model ... ')
        # self.w2v.wv.save_word2vec_format(root_path + '/model/embedding/w2v.bin', binary=False)

        logger.info('save fasttext model ... ')
        self.fast.wv.save_word2vec_format(root_path + '/model/embedding/fast.bin', binary=False)

        logger.info('save lda model ... ')
        # self.LDAmodel.save(root_path + '/model/embedding/lda')


    def load(self):
        logger.info('load tfidf model ... ')
        # self.tfidf = joblib.load(root_path + '/model/embedding/tfidf')

        logger.info('load w2v model ... ')
        # self.w2v = models.KeyedVectors.load_word2vec_format(root_path + '/model/embedding/w2v.bin', binary=False)

        logger.info('load fasttext model ... ')
        self.fast = models.KeyedVectors.load_word2vec_format(root_path + '/model/embedding/fast.bin', binary=False)

        logger.info('load lda model')
        # self.lda = LdaModel.load(root_path + '/model/embedding/lda')

        logger.info('load autoencoder model')
        # self.ae.load()


if __name__ == '__main__':
    em = Embedding()
    # em.load_data()
    # em.trainer()
    # em.saver()
    em.load()
    print(em.fast.wv)




