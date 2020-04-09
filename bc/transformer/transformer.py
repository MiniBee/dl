#!/usr/bin/env python
# _*_coding:utf-8 _*_
#@Time    :2020/3/12 下午9:57
#@Author  :hongyue pei 
#@FileName: transformer.py
#@Software: PyCharm

import tensorflow as tf
import numpy as np

try:
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass


class PositionEncoding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model):
        super(PositionEncoding, self).__init__()
        self.index = np.expand_dims(np.arange(0, d_model, 2), axis=0)
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)

    def call(self, input):
        max_sequence_len = input.shape[1]
        self.pe = np.zeros(shape=(max_sequence_len, self.d_model))
        self.position = np.expand_dims(np.arange(0, max_sequence_len), axis=1)
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
        # print('scaled dot product attention query shape ... ', query)
        # print('scaled dot product attention query matmul key shape ... ', query_matmul_key)
        scale = tf.sqrt(tf.cast(self.d_h, tf.float32))
        scaled_attention_score = query_matmul_key / scale
        if mask is not None:
            scaled_attention_score += mask * -1e9
        attention_weight = tf.nn.softmax(scaled_attention_score, axis=-1)

        return tf.matmul(attention_weight, value)


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, multi_head_count, d_model, dropout_prob):
        super(MultiHeadAttention, self).__init__()
        self.multi_head_count = multi_head_count
        self.d_model = d_model

        if d_model % multi_head_count != 0:
            raise ValueError('d_model({d_model}) % multi_head_count({multi_head_count}) != 0')

        self.d_h = d_model // multi_head_count
        self.linear_q = tf.keras.layers.Dense(d_model)
        self.linear_k = tf.keras.layers.Dense(d_model)
        self.linear_v = tf.keras.layers.Dense(d_model)
        self.scaled_dot_product_action = ScaledDotProductAction(self.d_h)
        self.linear = tf.keras.layers.Dense(d_model)

    def call(self, query, key, value, mask=None):
        # print('multi-head attention query shape ... ', query)
        query = self.linear_q(query)
        key = self.linear_k(key)
        value = self.linear_v(value)
        batch_size = tf.shape(query)[0]

        # print('multi-head attention split query shape ... ', query)
        # print('multi-head attention mask shape ... ', mask)

        query = self.split_head(query, batch_size)
        key = self.split_head(key, batch_size)
        value = self.split_head(value, batch_size)

        out = self.scaled_dot_product_action(query, value, key, mask)
        out = self.concat_head(out, batch_size)
        return self.linear(out)

    def split_head(self, tensor, batch_size):
        # print('multi-head attention split head tensor ... ', tensor)
        # print(tf.reshape(tensor, (batch_size, -1, 2, 512)))
        return tf.transpose(
            tf.reshape(
                tensor=tensor,
                shape=(batch_size, -1, self.multi_head_count, self.d_h)
                # tensor: (batch_size, seq_len_splited, attention_head_count, d_h)
            ),
            [0, 2, 1, 3]
            # tensor: (batch_size, attention_head_count, seq_len_splited, d_h)
        )

    def concat_head(self, tensor, batch_size):
        return tf.reshape(
            tensor=tf.transpose(tensor, [0, 2, 1, 3]),
            shape=(batch_size, -1, self.multi_head_count * self.d_h)
        )


class PositionWiseFeedForward(tf.keras.layers.Layer):
    def __init__(self, d_point_wise_ff, d_model):
        super(PositionWiseFeedForward, self).__init__()
        self.linear1 = tf.keras.layers.Dense(d_point_wise_ff)
        self.linear2 = tf.keras.layers.Dense(d_model)
        self.relu = tf.keras.layers.ReLU()

    def call(self, input):
        out = self.linear1(input)
        out = self.relu(out)
        out = self.linear2(out)
        return tf.add(input, out)


class Encoder(tf.keras.layers.Layer):
    def __init__(self, multi_head_count, d_model, d_point_wise_ff, dropout_prob):
        super(Encoder, self).__init__()
        self.multi_head_count = multi_head_count
        self.d_model = d_model
        self.d_point_wise_ff = d_point_wise_ff
        self.dropout1 = tf.keras.layers.Dropout(dropout_prob)
        self.layer_norm1 = tf.keras.layers.LayerNormalization()

        self.position_wise_fead_forward_layer = PositionWiseFeedForward(d_point_wise_ff, d_model)

        self.multi_head_attention = MultiHeadAttention(multi_head_count, d_model, dropout_prob)
        self.dropout2 = tf.keras.layers.Dropout(dropout_prob)
        self.layer_norm2 = tf.keras.layers.LayerNormalization()


    def call(self, x, mask=None, training=True):
        out = self.multi_head_attention(x, x, x, mask)
        out = self.dropout1(out, training=training)
        out = tf.add(x, out)
        print('encoder 143 out shape ', out)
        out = self.layer_norm1(out)
        out = self.position_wise_fead_forward_layer(out)
        out = self.dropout2(out)
        out = self.layer_norm2(out)
        return out


class Transformer(tf.keras.Model):
    def __init__(self, input_vocat_size, encoder_count, attention_head_count, d_model, d_point_wise_ff, dropout_prob, batch_size):
        super(Transformer, self).__init__()
        self.encoder_count = encoder_count
        self.attention_head_count = attention_head_count
        self.d_model = d_model
        self.d_point_wise_ff = d_point_wise_ff
        self.dropout_prob = dropout_prob
        self.batch_size = batch_size

        self.encoder_embedding_layer = PositionEncoding(input_vocat_size, self.d_model)
        self.encoder_embedding_dropout = tf.keras.layers.Dropout(dropout_prob)
        self.encoder_layers = [Encoder(attention_head_count, d_model, d_point_wise_ff, dropout_prob) for _ in range(encoder_count)]
        self.flatten = tf.keras.layers.Flatten()
        self.linear1 = tf.keras.layers.Dense(256, activation='tanh')
        self.linear2 = tf.keras.layers.Dense(1, activation='sigmoid')

    @tf.function
    def make_mask(self, input_):
        input = input_[:, 1:]
        mask = tf.cast(tf.math.equal(input, 0.), dtype=tf.float32)
        # sen_len = input_[:, 1]
        # mask = []
        # for i in range(self.batch_size):
        #     mask.append([0. if j < sen_len[i] else 1. for j in range(5000)])
        # mask = np.array(mask)
        # print(mask[:, tf.newaxis, tf.newaxis, :])
        # mask = tf.convert_to_tensor(mask, dtype=tf.float32)
        return input, mask[:, tf.newaxis, tf.newaxis, :]

    def call(self, input_, training=True):
        input, mask = self.make_mask(input_)
        encoder_tensor = self.encoder_embedding_layer(input)
        # print('transformer input shape ... ' * 50, encoder_tensor)
        # print(tf.reshape(encoder_tensor, (None, -1, 2, 256)))
        encoder_tensor = self.encoder_embedding_dropout(encoder_tensor)
        # print('encoder embedding shape ... ', encoder_tensor)
        for i in range(len(self.encoder_layers)):
            encoder_tensor = self.encoder_layers[i](encoder_tensor, mask, training=training)
        encoder_tensor = self.flatten(encoder_tensor)
        encoder_tensor = self.linear1(encoder_tensor)
        return self.linear2(encoder_tensor)


if __name__ == '__main__':
    PositionEncoding(10, 500).call(np.array([[23, 13], [1, 3]]))

