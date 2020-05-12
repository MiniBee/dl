#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: lstm.py
# @time: 2020/5/12 下午2:57
# @desc:

import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np

N_HIS = 3
# 'A_0', 'A_1', 'A_10', 'A_100', 'A_101', 'A_102', 'A_103', 'A_104', 'A_105', 'A_106', 'A_107', 'A_108', 'A_109', 'A_11', 'A_110', 'A_111', 'A_112', 'A_113', 'A_114', 'A_115', 'A_116', 'A_117', 'A_12', 'A_13', 'A_14', 'A_15', 'A_16', 'A_17', 'A_18', 'A_19', 'A_2', 'A_20', 'A_21', 'A_22', 'A_23', 'A_24', 'A_25', 'A_26', 'A_27', 'A_28', 'A_29', 'A_3', 'A_30', 'A_31', 'A_32', 'A_33', 'A_34', 'A_35', 'A_36', 'A_37', 'A_38', 'A_39', 'A_4', 'A_40', 'A_41', 'A_42', 'A_43', 'A_44', 'A_45', 'A_46', 'A_47', 'A_48', 'A_49', 'A_5', 'A_50', 'A_51', 'A_52', 'A_53', 'A_54', 'A_55', 'A_56', 'A_57', 'A_58', 'A_59', 'A_6', 'A_60', 'A_61', 'A_62', 'A_63', 'A_64', 'A_65', 'A_66', 'A_67', 'A_68', 'A_69', 'A_7', 'A_70', 'A_71', 'A_72', 'A_73', 'A_74', 'A_75', 'A_76', 'A_77', 'A_78', 'A_79', 'A_8', 'A_80', 'A_81', 'A_82', 'A_83', 'A_84', 'A_85', 'A_86', 'A_87', 'A_88', 'A_89', 'A_9', 'A_90', 'A_91', 'A_92', 'A_93', 'A_94', 'A_95', 'A_96', 'A_97', 'A_98', 'A_99', 'B_0', 'B_1', 'B_10', 'B_11', 'B_12', 'B_13', 'B_14', 'B_15', 'B_16', 'B_17', 'B_18', 'B_19', 'B_2', 'B_20', 'B_21', 'B_22', 'B_23', 'B_24', 'B_25', 'B_26', 'B_27', 'B_28', 'B_29', 'B_3', 'B_4', 'B_5', 'B_6', 'B_7', 'B_8', 'B_9', 'C_0', 'C_1', 'C_10', 'C_100', 'C_101', 'C_102', 'C_103', 'C_104', 'C_105', 'C_106', 'C_107', 'C_108', 'C_109', 'C_11', 'C_110', 'C_111', 'C_112', 'C_113', 'C_114', 'C_115', 'C_116', 'C_117', 'C_118', 'C_119', 'C_12', 'C_120', 'C_121', 'C_122', 'C_123', 'C_124', 'C_125', 'C_126', 'C_127', 'C_128', 'C_129', 'C_13', 'C_130', 'C_131', 'C_132', 'C_133', 'C_134', 'C_14', 'C_15', 'C_16', 'C_17', 'C_18', 'C_2', 'C_20', 'C_21', 'C_22', 'C_23', 'C_24', 'C_25', 'C_26', 'C_27', 'C_28', 'C_29', 'C_30', 'C_31', 'C_32', 'C_33', 'C_34', 'C_35', 'C_36', 'C_37', 'C_38', 'C_39', 'C_4', 'C_40', 'C_41', 'C_42', 'C_43', 'C_44', 'C_45', 'C_46', 'C_47', 'C_48', 'C_49', 'C_5', 'C_50', 'C_51', 'C_52', 'C_53', 'C_54', 'C_55', 'C_56', 'C_57', 'C_58', 'C_59', 'C_6', 'C_60', 'C_61', 'C_62', 'C_63', 'C_64', 'C_65', 'C_66', 'C_67', 'C_68', 'C_69', 'C_7', 'C_70', 'C_71', 'C_72', 'C_73', 'C_74', 'C_75', 'C_76', 'C_77', 'C_78', 'C_79', 'C_8', 'C_80', 'C_81', 'C_82', 'C_83', 'C_84', 'C_85', 'C_86', 'C_87', 'C_88', 'C_89', 'C_9', 'C_90', 'C_91', 'C_92', 'C_93', 'C_94', 'C_95', 'C_96', 'C_97', 'C_98', 'C_99', 'D_0', 'D_1', 'D_10', 'D_11', 'D_12', 'D_13', 'D_14', 'D_15', 'D_16', 'D_17', 'D_18', 'D_19', 'D_2', 'D_20', 'D_21', 'D_22', 'D_23', 'D_24', 'D_25', 'D_26', 'D_27', 'D_28', 'D_29', 'D_3', 'D_30', 'D_31', 'D_32', 'D_33', 'D_34', 'D_35', 'D_36', 'D_37', 'D_38', 'D_39', 'D_4', 'D_40', 'D_41', 'D_42', 'D_43', 'D_44', 'D_45', 'D_46', 'D_47', 'D_48', 'D_49', 'D_5', 'D_50', 'D_51', 'D_52', 'D_53', 'D_54', 'D_55', 'D_56', 'D_57', 'D_58', 'D_59', 'D_6', 'D_60', 'D_61', 'D_62', 'D_63', 'D_64', 'D_65', 'D_66', 'D_67', 'D_68', 'D_69', 'D_7', 'D_70', 'D_71', 'D_72', 'D_73', 'D_74', 'D_8', 'D_9', 'E_0', 'E_1', 'E_10', 'E_11', 'E_12', 'E_13', 'E_14', 'E_15', 'E_16', 'E_17', 'E_18', 'E_19', 'E_2', 'E_20', 'E_21', 'E_22', 'E_23', 'E_24', 'E_25', 'E_26', 'E_27', 'E_28', 'E_29', 'E_3', 'E_30', 'E_32', 'E_33', 'E_5', 'E_6', 'E_7', 'E_8', 'E_9'


def data_process1(mig_file, inf_file):
    x = []
    y = []
    m_x = []
    m_y = []
    i_y = []
    with open(mig_file) as f1, open(inf_file) as f2:
        for i, line in enumerate(f1):
            if i == 0:
                continue
            line_list = line.split(',')
            line_list = [float(x.strip()) for x in line_list[1:]]
            _max, _min = max(line_list), min(line_list)
            line_list = [(x-_min) / (_max - _min) for x in line_list]
            x.append(line_list)
        for i, line in enumerate(f2):
            if i == 0:
                continue
            line_list = line.split(',')
            line_list = [float(x.strip())/100 for x in line_list[1:]]
            y.append(line_list)
    for i, d in enumerate(x):
        if i >= N_HIS:
            m_x.append(x[i - N_HIS: i])
            m_y.append(x[i])
            i_y.append(y[i])

    return np.array(m_x), np.array(m_y), np.array(i_y), np.array(x)

class Model():
    def __init__(self, n_his, hidden_num, region_nums, loss, batch_size=1, epochs=5):
        self.n_his = n_his
        self.hidden_num = hidden_num
        self.region_nums = region_nums
        self.loss = loss
        self.batch_size = batch_size
        self.epochs = epochs

    def build_model(self):
        inputs = tf.keras.layers.Input(shape=(self.n_his, self.region_nums))
        outputs = tf.keras.layers.GRU(self.hidden_num)(inputs)
        # outputs = tf.keras.layers.LayerNormalization()(outputs)
        outputs = tf.keras.layers.Dropout(0.3)(outputs)
        outputs = tf.keras.layers.Dense(self.region_nums, activation='relu')(outputs)
        # outputs = tf.keras.layers.Dense(self.region_nums)(outputs)
        model = tf.keras.Model(inputs, outputs)
        model.compile(loss=self.loss, optimizer='adam')
        model.summary()
        return model

    def train(self, x_train, y_train, x_test, y_test, save_path, logpath):
        callbacks = [
            tf.keras.callbacks.TensorBoard(logpath),
            tf.keras.callbacks.ModelCheckpoint(save_path, save_best_only=True)
        ]
        self.model = self.build_model()
        self.model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.epochs, validation_data=(x_test, y_test), callbacks=callbacks)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = tf.keras.models.load_model(path)


def to_submit(ret_dict, path, path2):
    with open(path) as f1, open(path2, 'w') as f2:
        for line in f1:
            line = line.split(',')
            key = ','.join(line[:-1])
            w = key + ',' + str(int(ret_dict.get(key))) or '0'
            f2.write(w + '\n')


if __name__ == '__main__':
    mx, my, iy, x = data_process1('/Users/peihongyue/phy/project/dl/data/ai_studio/region_migration.csv', '/Users/peihongyue/phy/project/dl/data/ai_studio/infection.csv')
    mx_train, mx_test, my_train, my_test = train_test_split(mx, my, test_size=0.3)
    mx_train, mx_test, iy_train, iy_test = train_test_split(mx, iy, test_size=0.3)

    m1 = Model(N_HIS, 32, 388, 'mse', batch_size=32, epochs=100)
    m1.train(mx_train, my_train, mx_test, my_test, '/Users/peihongyue/phy/project/dl/data/ai_studio/best_m1.h5', '/Users/peihongyue/phy/project/dl/data/ai_studio/m1.log')
    m1 = tf.keras.models.load_model('/Users/peihongyue/phy/project/dl/data/ai_studio/best_m1.h5')


    m2 = Model(N_HIS, 32, 388, loss=tf.keras.losses.mean_squared_logarithmic_error, batch_size=32, epochs=200)
    m2.train(mx_train, iy_train, mx_test, iy_test, '/Users/peihongyue/phy/project/dl/data/ai_studio/best_m2.h5', '/Users/peihongyue/phy/project/dl/data/ai_studio/m2.log')
    m2 = tf.keras.models.load_model('/Users/peihongyue/phy/project/dl/data/ai_studio/best_m2.h5')
    idx = ['21200615', '21200616', '21200617', '21200618', '21200619', '21200620', '21200621', '21200622', '21200623', '21200624', '21200625', '21200626', '21200627', '21200628', '21200629', '21200630', '21200701', '21200702', '21200703', '21200704', '21200705', '21200706', '21200707', '21200708', '21200709', '21200710', '21200711', '21200712', '21200713', '21200714']
    region = ['A_0', 'A_1', 'A_10', 'A_100', 'A_101', 'A_102', 'A_103', 'A_104', 'A_105', 'A_106', 'A_107', 'A_108', 'A_109', 'A_11', 'A_110', 'A_111', 'A_112', 'A_113', 'A_114', 'A_115', 'A_116', 'A_117', 'A_12', 'A_13', 'A_14', 'A_15', 'A_16', 'A_17', 'A_18', 'A_19', 'A_2', 'A_20', 'A_21', 'A_22', 'A_23', 'A_24', 'A_25', 'A_26', 'A_27', 'A_28', 'A_29', 'A_3', 'A_30', 'A_31', 'A_32', 'A_33', 'A_34', 'A_35', 'A_36', 'A_37', 'A_38', 'A_39', 'A_4', 'A_40', 'A_41', 'A_42', 'A_43', 'A_44', 'A_45', 'A_46', 'A_47', 'A_48', 'A_49', 'A_5', 'A_50', 'A_51', 'A_52', 'A_53', 'A_54', 'A_55', 'A_56', 'A_57', 'A_58', 'A_59', 'A_6', 'A_60', 'A_61', 'A_62', 'A_63', 'A_64', 'A_65', 'A_66', 'A_67', 'A_68', 'A_69', 'A_7', 'A_70', 'A_71', 'A_72', 'A_73', 'A_74', 'A_75', 'A_76', 'A_77', 'A_78', 'A_79', 'A_8', 'A_80', 'A_81', 'A_82', 'A_83', 'A_84', 'A_85', 'A_86', 'A_87', 'A_88', 'A_89', 'A_9', 'A_90', 'A_91', 'A_92', 'A_93', 'A_94', 'A_95', 'A_96', 'A_97', 'A_98', 'A_99', 'B_0', 'B_1', 'B_10', 'B_11', 'B_12', 'B_13', 'B_14', 'B_15', 'B_16', 'B_17', 'B_18', 'B_19', 'B_2', 'B_20', 'B_21', 'B_22', 'B_23', 'B_24', 'B_25', 'B_26', 'B_27', 'B_28', 'B_29', 'B_3', 'B_4', 'B_5', 'B_6', 'B_7', 'B_8', 'B_9', 'C_0', 'C_1', 'C_10', 'C_100', 'C_101', 'C_102', 'C_103', 'C_104', 'C_105', 'C_106', 'C_107', 'C_108', 'C_109', 'C_11', 'C_110', 'C_111', 'C_112', 'C_113', 'C_114', 'C_115', 'C_116', 'C_117', 'C_118', 'C_119', 'C_12', 'C_120', 'C_121', 'C_122', 'C_123', 'C_124', 'C_125', 'C_126', 'C_127', 'C_128', 'C_129', 'C_13', 'C_130', 'C_131', 'C_132', 'C_133', 'C_134', 'C_14', 'C_15', 'C_16', 'C_17', 'C_18', 'C_2', 'C_20', 'C_21', 'C_22', 'C_23', 'C_24', 'C_25', 'C_26', 'C_27', 'C_28', 'C_29', 'C_30', 'C_31', 'C_32', 'C_33', 'C_34', 'C_35', 'C_36', 'C_37', 'C_38', 'C_39', 'C_4', 'C_40', 'C_41', 'C_42', 'C_43', 'C_44', 'C_45', 'C_46', 'C_47', 'C_48', 'C_49', 'C_5', 'C_50', 'C_51', 'C_52', 'C_53', 'C_54', 'C_55', 'C_56', 'C_57', 'C_58', 'C_59', 'C_6', 'C_60', 'C_61', 'C_62', 'C_63', 'C_64', 'C_65', 'C_66', 'C_67', 'C_68', 'C_69', 'C_7', 'C_70', 'C_71', 'C_72', 'C_73', 'C_74', 'C_75', 'C_76', 'C_77', 'C_78', 'C_79', 'C_8', 'C_80', 'C_81', 'C_82', 'C_83', 'C_84', 'C_85', 'C_86', 'C_87', 'C_88', 'C_89', 'C_9', 'C_90', 'C_91', 'C_92', 'C_93', 'C_94', 'C_95', 'C_96', 'C_97', 'C_98', 'C_99', 'D_0', 'D_1', 'D_10', 'D_11', 'D_12', 'D_13', 'D_14', 'D_15', 'D_16', 'D_17', 'D_18', 'D_19', 'D_2', 'D_20', 'D_21', 'D_22', 'D_23', 'D_24', 'D_25', 'D_26', 'D_27', 'D_28', 'D_29', 'D_3', 'D_30', 'D_31', 'D_32', 'D_33', 'D_34', 'D_35', 'D_36', 'D_37', 'D_38', 'D_39', 'D_4', 'D_40', 'D_41', 'D_42', 'D_43', 'D_44', 'D_45', 'D_46', 'D_47', 'D_48', 'D_49', 'D_5', 'D_50', 'D_51', 'D_52', 'D_53', 'D_54', 'D_55', 'D_56', 'D_57', 'D_58', 'D_59', 'D_6', 'D_60', 'D_61', 'D_62', 'D_63', 'D_64', 'D_65', 'D_66', 'D_67', 'D_68', 'D_69', 'D_7', 'D_70', 'D_71', 'D_72', 'D_73', 'D_74', 'D_8', 'D_9', 'E_0', 'E_1', 'E_10', 'E_11', 'E_12', 'E_13', 'E_14', 'E_15', 'E_16', 'E_17', 'E_18', 'E_19', 'E_2', 'E_20', 'E_21', 'E_22', 'E_23', 'E_24', 'E_25', 'E_26', 'E_27', 'E_28', 'E_29', 'E_3', 'E_30', 'E_32', 'E_33', 'E_5', 'E_6', 'E_7', 'E_8', 'E_9']
    ret_dict = {}
    for i, id in enumerate(idx):
        print(id)
        y = m1.predict(x[-N_HIS:][np.newaxis, :, :])[0]
        iy = m2.predict(x[-N_HIS:][np.newaxis, :, :])[0]
        x = list(x)
        x.append(y)
        x = np.array(x)
        line1 = [100 * x for x in list(iy)]
        line2 = [id]
        line2.extend(list(y))
        print(line1)
        for j, r in enumerate(region):
            key = ','.join(r.split('_') + [id])
            ret_dict[key] = line1[j]

    # to_submit(ret_dict, '/Users/peihongyue/phy/project/dl/data/ai_studio/submission.csv', '/Users/peihongyue/phy/project/dl/data/ai_studio/submission1.csv')









