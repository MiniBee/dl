#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: han.py
# @time: 2020/4/20 上午9:46
# @desc:

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers
import pandas as pd
import jieba
import json
import sys
sys.path.append('/home/peihongyue/project/python/dl/')
sys.path.append('/Users/peihongyue/phy/project/dl')
import numpy as np

from nlp_util import data_flow
from nlp_util import pre_process
from text_classification import attention2excel as a2e
# print(tf.__version__)

try:
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass


# text = ['she is a good student', 'xiaoming is good at basketball', 'he is Chinese']
# tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=50)
# tokenizer.fit_on_texts(text)
# # print(tokenizer.texts_to_sequences(text))
#
# tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer.to_json())
# print(tokenizer.texts_to_sequences(text))


def load_data():
    data = pd.read_csv('/Users/peihongyue/phy/project/dl/data/bc/bc.csv')
    data = data.fillna('')
    data['text'] = data.apply(lambda x: x['现病史（最近一次乳腺癌住院病历，后同）'] + x['诊疗过程描述'], axis=1)
    # data['text'] = data['text'].apply(lambda x: list(jieba.cut(str(x))))
    data['target'] = data['几线治疗'].apply(lambda x: 1 if x == '3' or x == 3 else 0)
    return data


def print3d(d3_array):
    for segment in d3_array:
        for i, sentence in enumerate(segment):
            print('sentence ' + str(i+1))
            print('$'.join([str(j) for j in sentence]))


def print2d(d2_array):
    for i, sentence in enumerate(d2_array):
        print('sentence ' + str(i+1))
        print('$'.join([str(j) for j in sentence]))
                

def get_token(text_list=None):
    if text_list:
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=9000)
        tokenizer.fit_on_texts(text_list)
        with open('/Users/peihongyue/phy/project/dl/data/bc/bc_tokenizer', 'w') as f:
            f.write(json.dumps(tokenizer.to_json()))
    else:
        with open('/Users/peihongyue/phy/project/dl/data/bc/bc_tokenizer') as f:
            line = f.readline()
            tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(json.loads(line))
    return tokenizer


class HierarchicalAttentionNetwork(tf.keras.layers.Layer):
    def __init__(self, attention_dim):
        super(HierarchicalAttentionNetwork, self).__init__()
        self.attention_dim = attention_dim
        self.dense1 = tf.keras.layers.Dense(self.attention_dim)
        self.activation1 = tf.keras.activations.tanh
        self.dense2 = tf.keras.layers.Dense(1)
        self.activation2 = tf.keras.activations.softmax


    def call(self, inputs):
        self.output1 = self.dense1(inputs)
        output1 = self.activation1(self.output1)
        output1 = self.dense2(output1)
        self.weight = self.activation2(output1)
        output1 = tf.reduce_mean(self.weight * inputs, axis=1)
        return output1

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

    def get_config():
        config = {"attention_dim":self.attention_dim}
        base_config = super(Mylayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class HAttention(tf.keras.layers.Layer):
    def __init__(self, regularizer=None, **kwargs):
        super(HAttention, self).__init__(**kwargs)
        self.regularizer = regularizer

    def build(self, input_shape):
        self.weight = self.add_weight(name='weight', shape=(input_shape[-1], 1), trainable=True, regularizer=self.regularizer)
        super(HAttention, self).build(input_shape)

    def call(self, inputs):
        attention_in = K.exp(K.squeeze(K.dot(inputs, self.weight), axis=-1))
        attention = attention_in/K.expand_dims(K.sum(attention_in, axis=-1), -1)
        weighted_sum = K.batch_dot(K.permute_dimensions(inputs, [0, 2, 1]), attention)
        return weighted_sum

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

    def get_config(self):
        base_config = super(HAttention, self).get_config()
        return dict(list(base_config.items()))



class HAN():
    def __init__(self, max_sentence, max_word):
        self.max_word = max_word
        self.max_sentence = max_sentence

    def han_model(self):
        l2_reg = regularizers.l2(1e-8)

        words = tf.keras.layers.Input(shape=(self.max_word,))
        embedded_words = tf.keras.layers.Embedding(10000, 512)(words)
        word_encoder1 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(100, return_sequences=True, kernel_regularizer=l2_reg))(embedded_words)
        word_encoder2 = tf.keras.layers.Dense(100, activation='tanh', name='dense_word_encoder1', kernel_regularizer=l2_reg)(word_encoder1)

        attention_weighted_words = tf.keras.Model(words, HAttention(name='word_attention', regularizer=l2_reg)(word_encoder2))
        self.word_attention_model = attention_weighted_words
        attention_weighted_words.summary()

        sentences = tf.keras.layers.Input(shape=(self.max_sentence, self.max_word))
        attention_weighted_sentences = tf.keras.layers.TimeDistributed(attention_weighted_words)(sentences)
        sentence_encoder1 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(50, return_sequences=True, kernel_regularizer=l2_reg))(attention_weighted_sentences)
        sentence_encoder2 = tf.keras.layers.Dense(100, activation='tanh', name='dense_sentence_encoder1', kernel_regularizer=l2_reg)(sentence_encoder1)
        attention_weighted_text = HAttention(name='sentence_attention', regularizer=l2_reg)(sentence_encoder2)
        predication = tf.keras.layers.Dense(1, activation='sigmoid')(attention_weighted_text)

        model = tf.keras.Model(sentences, predication)
        model.summary()
        model.compile(optimizer=tf.keras.optimizers.Adam(0.0004), loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
        return model


    def train(self, x_train, y_train, x_test, y_test):
        batch_size = 16
        epochs = 20
        self.model = self.han_model()
        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))


    def text2sentences(self, text):
        ret = []
        for sentences in text:
            for sentence in sentences:
                ret.append(sentence)
        return np.array(ret)


    def nopad_text(self, text):
        ret = []
        for sentences in text:
            sentence_list = []
            for words in sentences:
                if np.sum(words) == 0:
                    break
                sentence_list.append(words)
            if len(sentence_list) == 0:
                break
            ret.append(sentence_list)
        return np.array(ret)


    def save_model(self, path1, path2):
        self.word_attention_model.save(path1)
        self.model.save(path2)


    def load_model(self, path1, path2, custom_objects=None):
        self.word_attention_model = tf.keras.models.load_model(path1, custom_objects)
        self.model =  tf.keras.models.load_model(path2, custom_objects)


    def get_activations(self, text):
        sentences_from_text = self.text2sentences(text)
        word_encoder1_model = tf.keras.Model(inputs=self.word_attention_model.input, outputs = self.word_attention_model.get_layer('dense_word_encoder1').output)
        word_encoder1 = word_encoder1_model.predict(sentences_from_text)

        nopad_text = [self.nopad_text(i) for i in text]

        word_weights = self.word_attention_model.get_layer('word_attention').get_weights()[0]
        word_attention = sentences_from_text * np.exp(np.squeeze(np.dot(word_encoder1, word_weights)))

        word_attention = word_attention / np.expand_dims(np.sum(word_attention, -1), -1)
        nopad_attention = [list(filter(lambda x: x > 0, i)) for i in word_attention]

        # print2d(nopad_attention)

        sentence_encoder1_model = tf.keras.Model(inputs=self.model.input, outputs=self.model.get_layer('dense_sentence_encoder1').output)
        sentence_encoder1 = sentence_encoder1_model.predict(text)

        sentences_weights = self.model.get_layer('sentence_attention').get_weights()[0]
        sentences_attention = np.exp(np.squeeze(np.dot(sentence_encoder1, sentences_weights)))
        sentences_attention = sentences_attention / np.expand_dims(np.sum(sentences_attention, -1), -1)
        # print2d(sentences_attention)

        activation_map = []
        i = 0
        for k, segment in enumerate(text):
            sentence_att_list = []
            for j, sentance in enumerate(segment):
                sentence_att_list.append((sentences_attention[k][j], list(nopad_attention)[i]))
                i += 1
            activation_map.append(sentence_att_list)
        
        return activation_map


def get_origin_text(text, idx_path):
    word_idx = []
    segment_list = []
    with open(idx_path) as f:
        for line in f:
            word_idx.append(line.strip())
    for segment in text:
        sentences = []
        for sentence in segment:
            words = []
            for word in sentence:
                if word > 0:
                    words.append(word_idx[int(word) - 1])
            sentences.append(''.join(words))
        segment_list.append('。'.join(sentences))
    return segment_list


def get_word_from_idx(idx_path='/home/peihongyue/project/python/dl/data/bc/bc_word.idx'):
    word_idx = []
    with open(idx_path) as f:
        for line in f:
            word_idx.append(line.strip())
    return word_idx


def text_map_attention(text, activation):
    article_with_attention = []
    word_idx = get_word_from_idx('/home/peihongyue/project/python/dl/data/has_ae_idx.csv')
    for i, segment in enumerate(activation):  # idx of article
        sentences_attention = []
        for j, sentence in enumerate(segment):
            a = list(filter(lambda x: x > 0, text[i][j]))
            words_attention = list(zip(sentence[1], [word_idx[int(k) - 1] for k in a]))
            sentences_attention.append((sentence[0], words_attention))
        article_with_attention.append(sentences_attention)
    return article_with_attention


def load_data(path, max_sentence, max_word):
    x = []
    y = []
    with open(path) as f:
        for line in f:
            data = np.zeros(shape=(max_sentence, max_word))
            sentence_list = line.split('$$$1$$$')
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
    han = HAN(max_sentence, max_word)
    model_path2 = '/home/peihongyue/project/python/dl/data/has_ae2.h5'
    model_path1 = '/home/peihongyue/project/python/dl/data/has_ae1.h5'
    
    han.train(x_train, y_train, x_test, y_test)
    han.save_model(model_path1, model_path2)

    han.load_model(model_path1, model_path2, custom_objects={'HAttention': HAttention})

    activation_map = han.get_activations(x_train)
    # origin_text_list = get_origin_text(x_train[:2], '/home/peihongyue/project/python/dl/data/bc/bc_word.idx')
    x, y = load_data('/home/peihongyue/project/python/dl/data/has_ae_xy.csv', max_sentence, max_word)
    article_with_attention = text_map_attention(x, activation_map)
    # target_value, article_with_attention, excel_path
    a2e.to_excel(list(y), article_with_attention, 'test.xlsx')
    # for article in article_with_attention:
    #     print('-' * 50)
    #     for sentence in article:
    #         print(sentence)


if __name__ == '__main__':
    main()










