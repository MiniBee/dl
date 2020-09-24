#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: feature.py
# @time: 2020/9/24 下午4:03
# @desc:

import numpy as np
import copy
import torchvision.transforms as transforms

from __init__ import *
from src.utils.tools import wam, format_data
from src.tuils import config
import pandas as pd
import joblib
import json
import string
from jieba.posseg import pseg
from PIL import Image


def get_autoencoder_feature(data, max_features, max_len, model, tokenizer=None):
    x, _ = format_data(data, max_features, max_len, tokenizer=tokenizer, shuffle=True)
    data_ae = pd.DataFrame(model.predict(X, batch_size=64).max(axis=1), columns=['ae' + str(i) for i in range(max_len)])
    return data_ae


def get_lda_feature(lda_model, document):
    topic_importances = lda_model.get_document_topics(document, mininum_probability=0)
    topic_importances = np.array(topic_importances)
    return topic_importances[:, 1]


def get_pretrain_embedding(text, tokenizer, model):
    text_dict = tokenizer.encode_plus(text, add_special_tokens=True, max_length=400, ad_to_max_length=True)
    input_ids, attention_mask, token_type_ids = text_dict['input_ids'], text_dict['attention_mask'], text['token_type_ids']
    _, res = model(input_ids.to(config.device), attention_mask=attention_mask.to(config.device), token_type_ids=token_type_ids.to(config.device))
    return res.detach().cpu().numpy()[0]


def get_embedding_features(data, tfidf, embedding_model):
    data['queryCutRMStopWords'] = data['queryCutRMStopWord'].apply(lambda x: ' '.join(x))
    tfidf_data = pd.DataFrame(tfidf.transform(data['queryCutRMStopWords'].tolist()).toarray())
    tfidf_data.columns = ['tfidf' + str(i) for i in range(tfidf_data.shape[1])]
    data['w2v'] = data['queryCutRMStopWord'].apply(lambda x: wam(x, embedding_model, aggregate=False))

    train = copy.deepcopy(data)
    labelNameToIndex = json.load(open(config.root_path + '/data/label2id.json', encoding='utf-8'))
    labelIndexToName = {v: k for k, v in labelNameToIndex.items()}
    w2v_label_embedding = np.array(embedding_model.wv.get_vector(labelIndexToName[key]) for key in labelIndexToName if labelIndexToName[key] in embedding_model.wv.vocab.keys())

    joblib.dump(w2v_label_embedding, config.root_path + '/data/w2v_label_embedding.pkl')
    train = generate_feature(train, w2v_label_embedding, model_name='w2v')
    return tfidf_data, train


ch2en = {
    '！': '!',
    '？': '?',
    '｡': '.',
    '（': '(',
    '）': ')',
    '，': ',',
    '：': ':',
    '；': ';',
    '｀': ','
}


def tag_part_of_speech(data):
    words = [tuple(x) for x in list(pseg.cut(data))]
    noun_count = len([w for w in words if w[1] in ('NN', 'NNP', 'NNPS', 'NNS')])
    adjective_count = len([w for w in words if w[1] in ('JJ', 'JJR', 'JJS')])
    verb_count = len([w for w in words if w[1] in ('VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ')])
    return noun_count, adjective_count, verb_count


def get_basic_features(df):
    df['text'] = df['title'] + df['desc']
    df['queryCut'] = df['queryCut'].progress_apply(lambda x: [i if i not in ch2en.keys() else ch2en[i] for i in x])
    df['length'] = df['queryCut'].progress_apply(lambda x: len(x))
    return df


def generate_feature(data, label_embedding, model_name='w2v'):
    data[model_name + '_label_mean'] = data[model_name].progress_apply(
        lambda x: Find_Label_embedding(x, label_embedding, method='mean'))
    data[model_name + 'label_max'] = data[model_name].progress_apply(
        lambda x: Find_Label_embedding(x, label_embedding, method='max'))
    data[model_name + '_mean'] = data[model_name].progress_apply(
        lambda x: np.mean(np.array(x), axis=0))
    data[model_name + '_max'] = data[model_name].progress_apply(
        lambda x: np.max(np.array(x), axis=0))

    data[model_name + '_win_2_mean'] = data[model_name].progress_apply(
        lambda x: Find_embedding_with_windows(x, 2, method='mean'))
    data[model_name + '_win_3_mean'] = data[model_name].progress_apply(
        lambda x: Find_embedding_with_windows(x, 3, method='mean'))
    data[model_name + '_win_4_mean'] = data[model_name].progress_apply(
        lambda x: Find_embedding_with_windows(x, 4, method='mean'))
    data[model_name + '_win_2_max'] = data[model_name].progress_apply(
        lambda x: Find_embedding_with_windows(x, 2, method='max'))
    data[model_name + '_win_3_max'] = data[model_name].progress_apply(
        lambda x: Find_embedding_with_windows(x, 3, method='max'))
    data[model_name + '_win_4_max'] = data[model_name].progress_apply(
        lambda x: Find_embedding_with_windows(x, 4, method='max'))
    return data


def Find_embedding_with_windows(embedding_matrix, window_size=2, method='mean'):
    result_list = []
    for k1 in range(len(embedding_matrix)):
        if int(k1 + window_size) > len(embedding_matrix):
            result_list.extend(
                np.mean(embedding_matrix[k1:], axis=0).reshape(1, 300))
        else:
            result_list.extend(
                np.mean(embedding_matrix[k1:k1 + window_size],
                        axis=0).reshape(1, 300))
    if method == 'mean':
        return np.mean(result_list, axis=0)
    else:
        return np.max(result_list, axis=0)


def softmax(x):
    return np.exp(x) / np.exp(x).sum(axis=0)


def Find_Label_embedding(example_matrix, label_embedding, method='mean'):
    similarity_matrix = np.dot(example_matrix, label_embedding.T) / (np.linalg.norm(example_matrix) * np.linalg.norm(label_embedding))
    attention = similarity_matrix.max()
    attention = softmax(attention)
    attention_embedding = attention * example_matrix
    if method == 'mean':
        return np.mean(attention_embedding, axis=0)
    else:
        return np.max(attention_embedding, axis=0)





