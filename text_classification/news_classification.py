#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: news_classification.py
# @time: 2020/5/8 下午20:46
# @desc:

import pandas as pd
import json
from sklearn.model_selection import train_test_split
import sys
sys.path.append('/home/peihongyue/project/python/dl/')

from nlp_util import data_flow, word2idx
from han import HAN


def pre_training(file_path):
    df = pd.read_csv(file_path + 'sqlResult_1558435.csv', encoding='gb18030', engine='python')
    df = df.fillna('')
    df['target'] = df['feature'].apply(lambda x: json.loads(x).get('type'))
    df['content'] = df.apply(lambda x: x['title'] + '。' + x['content'], axis=1)
    df['content'] = df['content'].apply(lambda x: str(x).replace('\n', '。').replace(' ', ''))
    df = df[['content', 'target']]
    content_list = df['content'].tolist()
    target_list = df['target'].tolist()
    content_word_list = word2idx.tokens(content_list, save_path=file_path + 'content_tokens.csv')
    word_idx = word2idx.word_set(content_word_list, save_path=file_path + 'idx.csv')
    content_idx = word2idx.word2idx(content_word_list, word_idx, save_path=file_path + 'news2idx.csv')
    x_train, x_test, y_train, y_test = train_test_split(content_idx, target_list, test_size=0.3)

    for i, line in enumerate(x_train):
        target_contect_list.append([y_train[i]] + x_train[i])
    word2idx.save_list(target_contect_list, file_path + '/news_train.csv')

    target_contect_list = []
    for i, line in enumerate(x_test):
        target_contect_list.append([y_test[i]] + x_test[i])
    word2idx.save_list(target_contect_list, file_path + '/news_test.csv')

def load_data(path, max_sentence, max_word):
    x = []
    y = []
    with open(path) as f:
        for line in f:
            data = np.zeros(shape=(max_sentence, max_word))
            sentence_list = line.split('$$$2$$$')
            for i, sentence in enumerate(sentence_list):
                word_list = sentence.split('$$$')
                if i == 0:
                    y.append(float(word_list[0]))
                    word_list = word_list[1:]
                if i >= max_sentence:
                    break
                for j, word in enumerate(word_list):
                    if j >= max_word:
                        break
                    data[i, j] = float(word)
            x.append(data)
    return np.array(x), np.array(y)

def main():
    max_sentence = 50
    max_word = 100
    try:
        x_train, y_train = load_data('/Users/peihongyue/phy/project/dl/data/has_ae_train.csv', max_sentence, max_word)
        x_test, y_test = load_data('/Users/peihongyue/phy/project/dl/data/bc/bc_test.csv', max_sentence, max_word)
    except:
        x_train, y_train = load_data('/home/peihongyue/project/python/dl/data/has_ae_train.csv', max_sentence, max_word)
        x_test, y_test = load_data('/home/peihongyue/project/python/dl/data/has_ae_test.csv', max_sentence, max_word)
    print('x_train shape: ', x_train.shape)
    print('y_train shape:', y_train.shape)
    han = HAN(max_sentence, max_word, 2)
    model_path2 = '/home/peihongyue/project/python/dl/data/has_ae2.h5'
    model_path1 = '/home/peihongyue/project/python/dl/data/has_ae1.h5'
    
    han.train(x_train, y_train, x_test, y_test)
    han.save_model(model_path1, model_path2)
    han.load_model(model_path1, model_path2, custom_objects={'HAttention': HAttention})
    activation_map = han.get_activations(x_train)
    x, y = load_data('/home/peihongyue/project/python/dl/data/has_ae_xy.csv', max_s
    
    
    
    
    
    entence, max_word)
    article_with_attention = text_map_attention(x, activation_map)
    a2e.to_excel(list(y), article_with_attention, 'test.xlsx')


if __name__ == "__main__":
    file_path = '/home/peihongyue/project/python/dl/data/news/'
    pre_training(file_path)
    


