#!/usr/bin/env python
# _*_coding:utf-8 _*_
#@Time    :2020/3/12 下午9:57
#@Author  :hongyue pei 
#@FileName: transformer.py
#@Software: PyCharm

import tensorflow as tf
import numpy as np
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)


class PositionEncoding(tf.keras.layers.Layer):
    def __init__(self, max_sequence_len, vocab_size, d_model):
        super(PositionEncoding, self).__init__()
        self.pe = np.zeros(shape=(max_sequence_len, d_model))
        self.position = np.expand_dims(np.arange(0, max_sequence_len), axis=1)
        self.index = np.expand_dims(np.arange(0, d_model, 2), axis=0)
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)

    def call(self, input):
        self.pe[:, 0::2] = np.sin(self.pos())
        self.pe[:, 1::2] = np.cos(self.pos())
        return self.pe + self.embedding(input)

    def pos(self):
        return self.position / (np.power(1000, (self.index - self.index % 2) / float(self.d_model)))


class ScaledDotProductAction(tf.keras.layers.Layer):
    def __init__(self, d_h):
        super(ScaledDotProductAction, self).__init__()
        self.d_h = d_h

    def call(self, query, value, key, mask=None):
        query_matmul_key = tf.matmul(query, key, transpose_b=True)
        scale = tf.sqrt(tf.cast(self.d_dim, tf.float32))
        scaled_attention_score = query_matmul_key / scale
        if mask is not None:
            scaled_attention_score += (mask * -1e9)
        attention_weight = tf.nn.softmax(scaled_attention_score)
        return attention_weight * value


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, multi_head_count, d_model, dropout_prob):
        super(MultiHeadAttention, self).__init__()
        self.multi_head_count = multi_head_count
        self.d_model = d_model

        if d_model % multi_head_count != 0:
            raise ValueError('d_model({d_model}) % multi_head_count({multi_head_count}) != 0')

        self.d_h = d_model / multi_head_count
        self.linear_q = tf.keras.layers.Dense(d_model)
        self.linear_k = tf.keras.layers.Dense(d_model)
        self.linear_v = tf.keras.layers.Dense(d_model)
        self.scaled_dot_product_action = ScaledDotProductAction(self.d_h)
        self.linear = tf.keras.layers.Dense(d_model)

    def call(self, query, key, value, mask=None):
        query = self.linear_q(query)
        key = self.linear_k(key)
        value = self.linear_v(value)
        batch_size = np.shape(query)[0]
        query = self.split_head(query, batch_size)
        key = self.split_head(key, batch_size)
        value = self.split_head(value, batch_size)

        out = self.scaled_dot_product_action(query, value, key, mask)
        out = self.concat_head(out, batch_size)
        return self.linear(out)

    def split_head(self, tensor, batch_size):
        return tf.transpose(tf.reshape(tensor, (batch_size, -1, self.multi_head_count, self.d_h)), [0, 2, 1, 3])

    def concat_head(self, tensor, batch_size):
        return tf.reshape(tf.transpose(tensor, [0, 2, 1, 3]), (batch_size, -1, self.multi_head_count * self.d_h))


class PositionWiseFeedForward(tf.keras.layers.Layer):
    def __init__(self, d_point_wise_ff, d_model):
        super(PositionWiseFeedForward, self).__init__()
        self.linear1 = tf.keras.layers.Dense(d_point_wise_ff)
        self.linear2 = tf.keras.layers.Dense(d_model)

    def call(self, input):
        out = self.linear1.Dense(input)
        out = tf.keras.layers.ReLU(out)
        out = self.linear2.Dense(out)
        return out + input


class Encoder(tf.keras.layers.Layer):
    def __init__(self, multi_head_count, d_model, d_point_wise_ff, dropout_prob):
        super(Encoder, self).__init__()
        self.multi_head_count = multi_head_count
        self.d_model = d_model
        self.d_point_wise_ff = d_point_wise_ff
        self.dropout1 = tf.keras.layers.Dropout(dropout_prob)
        self.layer_norm1 = tf.keras.layers.LayerNormalization()

        self.position_wise_fead_forward_layer = PositionWiseFeedForward(d_point_wise_ff, d_model, dropout_prob)

        self.multi_head_attention = MultiHeadAttention(multi_head_count, d_model, dropout_prob)
        self.dropout2 = tf.keras.layers.Dropout(dropout_prob)
        self.layer_norm2 = tf.keras.layers.LayerNormalization()

    def call(self, x, mask, traning):
        out = self.multi_head_attention(x, x, x, mask)
        out = self.dropout1(out, traning=traning)
        out = tf.add(x, out)
        out = self.layer_norm1(out)
        out = self.position_wise_fead_forward_layer(out)
        out = self.dropout2(out)
        out = self.layer_norm2(out)
        return out


class Transformer(tf.keras.Model):
    def __init__(self):
        super(Transformer, self).__init__()


if __name__ == '__main__':
    PositionEncoding(10, 500).call()

