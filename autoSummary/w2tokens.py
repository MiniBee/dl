#!/usr/bin/env python
# _*_coding:utf-8 _*_
#@Time    :2020/2/22 下午9:23
#@Author  :hongyue pei 
#@FileName: w2tokens.py
#@Software: PyCharm


import pandas as pd
import re
import jieba
from gensim import corpora
from gensim.models import word2vec
import data_flow
import os
import json


def token(string):
    return re.findall('\w+', string)


def read_file(file_path, file_dict):
    content = pd.read_csv(file_path, encoding='utf-8')
    return content


def _corpora(tokens_cut, dict_path, save):
    if save:
        dictionary = corpora.Dictionary(tokens_cut)
        dictionary.save(dict_path)
    else:
        dictionary = corpora.Dictionary()
        dictionary.load_from_text(dict_path)
    print(dictionary)


def _word2vec(tokens_cut):
    model = word2vec.Word2Vec(tokens_cut, min_count=1)
    model.wv.save_word2vec_format('/home/peihongyue/project/python/dl/data/autoSummary/word2vec')
    # model = word2vec.Word2Vec()
    # model = model.wv.load_word2vec_format('/home/peihongyue/project/python/dl/data/autoSummary/word2vec')


def cut_sen(content, dict_path, save):
    if save or not os.path.exists('/home/peihongyue/project/python/dl/data/autoSummary/tokens_cut'):
        articles = content['content'].apply(lambda x: str(x).replace('\r\n', '')).tolist()
        titles = content['title'].apply(lambda x: str(x).replace('\r\n', '')).tolist()
        articles_token = [str(a) for a in articles]
        titles_token = [str(a) for a in titles]
        tokens = ['。'.join([a[1], a[0]]) for a in list(zip(articles_token, titles_token))]
        tokens_cut = []
        for i, content_ls in enumerate([token(a) for a in tokens]):
            print(i)
            for c in content_ls:
                w = list(jieba.cut(c))
                tokens_cut.append(w)

        with open('/home/peihongyue/project/python/dl/data/autoSummary/tokens_cut', 'w') as f:
            for tc in tokens_cut:
                f.write('@_@'.join(tc) + '\n')
    else:
        tokens_cut = []
        with open('/home/peihongyue/project/python/dl/data/autoSummary/tokens_cut') as f:
            for line in f:
                line = line.replace('\n', '')
                tokens_cut.append(line.split('@_@'))
    data_flow.getWordFrequency(tokens_cut)
    print(tokens_cut[:3])
    # _word2vec(tokens_cut)


def main(contect):
    articles = content['content'].tolist()
    titles = content['title'].tolist()
    articles_cut = []



if __name__ == '__main__':
    file_path = '/home/peihongyue/data/kaikeba/zhwiki/zhwikidata/news.csv'
    file_dict = '/home/peihongyue/data/kaikeba/zhwiki/zhwikidata/data'
    dict_path = '/home/peihongyue/project/python/dl/data/autoSummary/token_dict.dict'
    content = read_file(file_path, file_dict)
    cut_sen(content, dict_path, save=False)



