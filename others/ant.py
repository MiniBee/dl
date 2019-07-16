#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: ant.py
# @time: 2019/7/16 上午11:22
# @desc:

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras import callbacks as kcallbacks
from keras.initializers import glorot_uniform
from keras.models import Sequential,Model,K
from keras.layers import Dense,Dropout,normalization
from keras.optimizers import SGD
import tensorflow as tf
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.externals import joblib
import os

np.random.seed(4)
tf.set_random_seed(13)
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

class Solver(object):
    def __init__(self,X_train,X_valid,y_train,y_valid, model_path, train=True):
        self.X_train = X_train
        self.X_valid = X_valid
        self.y_train = y_train
        self.y_valid = y_valid
        self.model_path = model_path
        self.model = None
        if train:
            self.Xgboost_classsifier()
        else:
            if os.path.exists(self.model_path):
              self.model = joblib.load(self.model_path)
            else:
                raise Exception('[!] No such file ... ' + self.model_path)

    def Xgboost_classsifier(self, max_depth=3, learning_rate=0.1, n_estimators=100):
        import xgboost as xgb
        xgb_model = xgb.XGBClassifier()
        param_grid = {'max_depth': [1,2,3,4], 'learning_rate': [0.1, 0.001, 0.005, 0.01], 'n_estimators':[100, 200, 300]}
        xgb_grid = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='accuracy', cv=10, n_jobs=5)
        xgb_grid.fit(self.X_train, self.y_train)
        self.model = xgb_grid.best_estimator_
        joblib.dump(self.model, self.model_path)

    def predict(self):
        if self.model:
            y_pred = self.model.predict(self.X_valid)
            return self.X_valid, y_pred
        else:
            raise Exception('[!] init model first ... ')

    # def reg_eval(self):
    #     pass
    #
    # def class_eval(Self):
    #     pass
    #
    # def mse(self, y, y_pred):
    #     return tf.reduce_mean(tf.square(y_pred - y))
    #
    # def cross_entropy(self, y, y_pred):
    #     m = y.shape[0]
    #     loss = 0.0
    #     for i in range(m):
    #         loss -= y[i] * np.log(y_pred[i].T)
    #     return loss / m
    #
    # def binary_cross_entropy(self, y, y_pred):
    #     m = y.shape[0]
    #     loss = 0.0
    #     for i in range(m):
    #         print(y[i], y_pred[i])
    #         loss -= y[i] * np.log(max([min([y_pred[i], 1]), 1e-10])) + (1-y[i])*np.log(max([min([1-y_pred[i], 1]), 1e-10]))
    #     return loss / m


class Nodes:
    def __init__(self):
        self.X_train = None
        self.X_valid = None
        self.y_train = None
        self.y_valid = None
        self.decision = None
        self.direct = None
        self.loss = None

class Auto_Grow_Tree(object):
    def __init__(self,X_train,X_valid,y_train,y_valid,X_test = None,y_test = None,optimizer = 'adam',num_n = 6,categories = 3,act = 'relu',drop_rate = 0,batch_size = 5,epochs = 100,patience = 5,solver = None):
        self.X_train = X_train
        self.X_valid = X_valid
        self.y_train = y_train
        self.y_valid = y_valid
        self.X_test = X_test
        self.y_test = y_test
        self.optimizer = optimizer
        self.num_n = num_n # neuron nums
        self.output_dim = categories # output_dim >= 2 作为参数传入output_dim（全局变量）#当output_dim = 1 时为regression 任务
        self.act = act
        self.drop_rate = drop_rate # drop_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience # for early stop
        self.Solver = Solver
        self.min_loss = np.inf
        self.node_dict = None
        self.predict_groups = None

    def basic_model(self,in_dim):
        model = Sequential()
        # Adding the input layer and the first hidden layer
        model.add(Dense(output_dim = self.num_n, init = glorot_uniform(seed=1), activation = self.act, input_dim = in_dim))
        model.add(normalization.BatchNormalization())
        if self.drop_rate > 0:
            model.add(Dropout(self.drop_rate,seed = 1))
        return model

    def deep_layer(self,model):
        model.add(Dense(output_dim = self.num_n, init = glorot_uniform(seed=1), activation = self.act))
        model.add(normalization.BatchNormalization())
        if self.drop_rate > 0:
            model.add(Dropout(self.drop_rate,seed = 1))
        return model

    def out_layer(self,model):
        if self.output_dim == 1:
            model.add(Dense(output_dim = 1, init = glorot_uniform(seed=1), activation = 'linear'))
        elif self.output_dim == 2 :
            model.add(Dense(output_dim = 1, init = glorot_uniform(seed=1), activation = 'sigmoid'))
        else:
            model.add(Dense(output_dim = self.output_dim, init = glorot_uniform(seed=1), activation = 'softmax'))
        return model

    def compile_model(self,model):
        if self.output_dim == 1:
            model.compile(optimizer =self.optimizer, loss =  'mean_squared_error')
        elif self.output_dim == 2 :
            model.compile(optimizer =self.optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
        else:
            model.compile(optimizer = self.optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
        return model

    def train_model(self,name,model,X_train,X_valid,y_train,y_valid):
        best_weights_filepath = './'+name+'_'+str(len(model.layers))+'.hdf5'
        earlyStopping=kcallbacks.EarlyStopping(monitor='val_loss', patience= self.patience, verbose=1, mode='auto')
        saveBestModel = kcallbacks.ModelCheckpoint(best_weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
        model.fit(X_train, y_train, batch_size= self.batch_size, epochs=self.epochs,verbose=1,validation_data = (X_valid,y_valid),callbacks=[earlyStopping, saveBestModel])
        model.load_weights(best_weights_filepath)
        return model

    def eval_model(self,model,X_valid,y_valid):   ### +with_solver
        y_pred = model.predict(X_valid)
        if self.output_dim == 1:
            y_pred = tf.reshape(tf.convert_to_tensor(y_pred, np.float32),[-1])
            y_valid = tf.reshape(tf.convert_to_tensor(y_valid, np.float32),[-1])
            return K.mean(keras.losses.mean_squared_error(y_valid, y_pred)).eval(session = tf.Session())
        elif self.output_dim == 2:
            y_pred = tf.reshape(tf.convert_to_tensor(y_pred, np.float32),[-1])
            y_valid = tf.reshape(tf.convert_to_tensor(y_valid, np.float32),[-1])
            return K.mean(K.binary_crossentropy(y_valid, y_pred), axis=-1).eval(session = tf.Session())
        else:
            y_pred = tf.reshape(tf.convert_to_tensor(y_pred, np.float32),[-1,self.output_dim])
            y_valid = tf.reshape(tf.convert_to_tensor(y_valid, np.float32),[-1,self.output_dim])
            return K.mean(K.categorical_crossentropy(y_valid, y_pred), axis=-1).eval(session = tf.Session())

    def new_build(self,name,X_train,X_valid,y_train,y_valid):
        model = self.basic_model(len(X_train[0]))
        model = self.out_layer(model)
        model = self.compile_model(model)
        self.train_model(name,model,X_train,X_valid,y_train,y_valid)
        loss_tmp = self.eval_model(model,X_valid,y_valid)
        return loss_tmp,model

    def parts_split(self,model,X,y,want = 'train'):
        dense_layer_model = Model(inputs=model.input,outputs=model.get_layer(index = -2).output)
        hidden = dense_layer_model.predict(X)
        y_pred = model.predict(X)
        mix = np.concatenate((X,hidden),1)
        if self.output_dim == 2:
            left_ind = np.where(y_pred <= 0.5)[0]
            right_ind = np.where(y_pred > 0.5)[0]
            if want == 'train':
                return [(X[left_ind],y[left_ind]),(X[right_ind],y[right_ind])]
            elif want == 'hidden': # hidden layer info
                return [(hidden[left_ind],y[left_ind]),(hidden[right_ind],y[right_ind])]
            elif want == 'mix': # for mix
                return [(mix[left_ind],y[left_ind]),(mix[right_ind],y[right_ind])]
            else :
                raise NameError
        else:
            out = []
            for i in range(self.output_dim):
                ind = np.where(np.argmax(y_pred,axis = 1) == i)[0]
                if want =='train':
                    out.append((X[ind],y[ind]))
                elif want == 'hidden':
                    out.append((hidden[ind],y[ind]))
                elif want == 'mix':
                    out.append((mix[ind],y[ind]))
                else :
                    raise NameError
            return out


    def sub_train(self,name,model,X_train,X_valid,y_train,y_valid,want):
        out = []
        for i in range(self.output_dim):
            X_train_tmp,y_train_tmp = self.parts_split(model,X_train,y_train,want)[i]
            X_valid_tmp,y_valid_tmp = self.parts_split(model,X_valid,y_valid,want)[i]
            out.append(self.new_build(name+'_'+str(i)+want,X_train_tmp,X_valid_tmp,y_train_tmp,y_valid_tmp))
        return out

    def eval_split(self,model_l,parts_valid):   ### + with solver
        y_pred_l = []
        y_valid_l = []
        for i in range(self.output_dim):
            tmp_model = model_l[i][1]
            X_valid_tmp,y_valid_tmp = parts_valid[i]
            y_pred_l.append(tmp_model.predict(X_valid_tmp))
            y_valid_l.append(y_valid_tmp)
        y_pred = np.concatenate(y_pred_l)
        y_valid = np.concatenate(y_valid_l)
        if self.output_dim == 2:
            y_pred = tf.reshape(tf.convert_to_tensor(y_pred, np.float32),[-1])
            y_valid = tf.reshape(tf.convert_to_tensor(y_valid, np.float32),[-1])
            return K.mean(K.binary_crossentropy(y_valid, y_pred), axis=-1).eval(session = tf.Session())
        else:
            y_pred = tf.reshape(tf.convert_to_tensor(y_pred, np.float32),[-1,self.output_dim])
            y_valid = tf.reshape(tf.convert_to_tensor(y_valid, np.float32),[-1,self.output_dim])
            return K.mean(K.categorical_crossentropy(y_valid, y_pred), axis=-1).eval(session = tf.Session())

    def try_split(self,name,model,X_train,X_valid,y_train,y_valid):
        model_train_l = self.sub_train(name,model,X_train,X_valid,y_train,y_valid,'train')
        model_hidden_l = self.sub_train(name,model,X_train,X_valid,y_train,y_valid,'hidden')
        model_mix_l = self.sub_train(name,model,X_train,X_valid,y_train,y_valid,'mix')
        parts_train_valid = self.parts_split(model,X_valid,y_valid,want = 'train')
        parts_hidden_valid = self.parts_split(model,X_valid,y_valid,want = 'hidden')
        parts_mix_valid = self.parts_split(model,X_valid,y_valid,want = 'mix')
        loss1 = self.eval_split(model_train_l,parts_train_valid)
        loss2 = self.eval_split(model_hidden_l,parts_hidden_valid)
        loss3 = self.eval_split(model_mix_l,parts_mix_valid)
        loss_min = min(loss1,loss2,loss3)
        if loss1 == loss_min:
            loss_parts = [x[0] for x in model_train_l]
            model_parts = [x[1] for x in model_train_l]
            want = 'train'
        if loss2 == loss_min:
            loss_parts = [x[0] for x in model_hidden_l]
            model_parts = [x[1] for x in model_hidden_l]
            want = 'hidden'
        if loss3 == loss_min:
            loss_parts = [x[0] for x in model_mix_l]
            model_parts = [x[1] for x in model_mix_l]
            want = 'mix'
        return loss_min,model_parts,loss_parts,want

    def try_deep(self,name,model,X_train,X_valid,y_train,y_valid):
        model.pop()
        model = self.deep_layer(model)
        model = self.out_layer(model)
        model = self.compile_model(model)
        model = self.train_model(name,model,X_train,X_valid,y_train,y_valid)
        result = self.eval_model(model,X_valid,y_valid)
        return result,model

    def get_decision(self,name,result_old,model,X_train,X_valid,y_train,y_valid):
        if self.output_dim > 1:
            X_list_size = min([x[1].shape[0] for x in self.parts_split(model,X_train,y_train)])
            X_list_size2 = min([x[1].shape[0] for x in self.parts_split(model,X_valid,y_valid)])
            if min(X_list_size,X_list_size2) < self.batch_size:
                result_split = np.inf
            else:
                result_split,model_parts,loss_parts,split_type = self.try_split(name,model,X_train,X_valid,y_train,y_valid)
        else:
            result_split = np.inf
        model.pop()
        model = self.deep_layer(model)
        model = self.out_layer(model)
        model = self.compile_model(model)
        self.train_model(name,model,X_train,X_valid,y_train,y_valid)
        result_new = self.eval_model(model,X_valid,y_valid)
        min_result = min(result_old*0.99,result_new,result_split)
        if result_old*0.99 == min_result:
            model.pop()
            model.pop()
            model.pop()
            model = self.out_layer(model)
            model = self.compile_model(model)
            model.load_weights(name+'_'+str(len(model.layers))+'.hdf5')
            return min_result,model,None,None,'keep'
        elif result_new == min_result:
            model.load_weights(name+'_'+str(len(model.layers))+'.hdf5')
            return min_result,model,None,None,'deep'
        else:
            model.pop()
            model.pop()
            model.pop()
            model = self.out_layer(model)
            model= self.compile_model(model)
            model.load_weights(name+'_'+str(len(model.layers))+'.hdf5')
            for i in range(self.output_dim):
                model_parts[i].save_weights(name+'_'+str(i)+'_'+str(len(model_parts[i].layers))+'.hdf5')
            return min_result,model,loss_parts,model_parts,split_type

    def auto_grow(self):
        node_dict = {}
        wait_list = ['root']
        root = Nodes()
        root.X_train,root.X_valid,root.y_train,root.y_valid = self.X_train,self.X_valid,self.y_train,self.y_valid
        wait_dict = {'root': root}
        while wait_list:
            tmp = wait_list.pop(0)
            X_train,X_valid,y_train,y_valid = wait_dict[tmp].X_train,wait_dict[tmp].X_valid,wait_dict[tmp].y_train,wait_dict[tmp].y_valid
            if not wait_dict[tmp].direct:
                loss_tmp,model = self.new_build(tmp,X_train,X_valid,y_train,y_valid)
            else:
                loss_tmp,model = wait_dict[tmp].loss,wait_dict[tmp].direct
            min_result,direct_model,loss_l,model_parts,decision = self.get_decision(tmp,loss_tmp,model,X_train,X_valid,y_train,y_valid)
            if decision == 'keep':
                node_dict[tmp] = Nodes()
                node_dict[tmp].direct = direct_model
                node_dict[tmp].loss = min_result
                continue
            elif decision == 'deep':
                wait_list = [tmp] + wait_list
                wait_dict[tmp].direct = direct_model
                wait_dict[tmp].loss = min_result
                continue
            else:
                node_dict[tmp] = Nodes()
                node_dict[tmp].direct =  direct_model
                node_dict[tmp].loss = min_result
                node_dict[tmp].decision = decision
                list_Nodes = [tmp+'_'+str(i) for i in range(self.output_dim)]
                wait_list += list_Nodes
                for i in range(self.output_dim):
                    tmp_tmp = list_Nodes[i]
                    wait_dict[tmp_tmp] = Nodes()
                    wait_dict[tmp_tmp].loss = loss_l[i]
                    X_train_part,y_train_part = self.parts_split(model,X_train,y_train,decision)[i]
                    X_valid_part,y_valid_part = self.parts_split(model,X_valid,y_valid,decision)[i]
                    wait_dict[tmp_tmp].X_train,wait_dict[tmp_tmp].X_valid,wait_dict[tmp_tmp].y_train,wait_dict[tmp_tmp].y_valid = X_train_part,X_valid_part,y_train_part,y_valid_part
        self.node_dict = node_dict
        return node_dict

    def predict_flow(self,name,X_test,y_test):
        if y_test.size == 0 :
            return X_test,y_test
        model = self.node_dict[name].direct
        decision = self.node_dict[name].decision
        if not decision :
            y_pred = model.predict(X_test)
            return y_pred,y_test
        return self.parts_split(model,X_test,y_test,want = decision)

    def final_predict(self):
        out_dict = {}
        wait_list = ['root']
        wait_dict = {'root': (self.X_test,self.y_test)}
        while wait_list:
            name = wait_list.pop(0)
            if name not in node_dict:
                continue
            X_test,y_test = wait_dict[name]
            if not self.node_dict[name].decision :
                out_dict[name] = self.predict_flow(name,X_test,y_test)
            else:
                for i in range(self.output_dim):
                    wait_list += [name+'_'+str(i)]
                    wait_list[name+'_'+str(i)] = self.predict_flow(name,X_test,y_test)[i]
        self.predict_groups = out_dict
        return out_dict

# if __name__ == '__main__':
#
#     # Importing the dataset
#     dataset = pd.read_csv('Churn_Modelling.csv')
#     X = dataset.iloc[:, 3:13].values
#     y = dataset.iloc[:, 13].values
#
#     # Encoding categorical data
#     from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#     labelencoder_X_1 = LabelEncoder()
#     X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
#     labelencoder_X_2 = LabelEncoder()
#     X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
#     onehotencoder = OneHotEncoder(categorical_features = [1])
#     X = onehotencoder.fit_transform(X).toarray()
#     X = X[:, 1:]
#
#     # Splitting the dataset into the Training set and Test set
#     from sklearn.model_selection import train_test_split
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
#     X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.125, random_state = 0)
#
#     # Feature Scaling
#     from sklearn.preprocessing import StandardScaler
#     sc = StandardScaler()
#     X_train = sc.fit_transform(X_train)
#     X_valid = sc.fit_transform(X_valid)
#     X_test = sc.transform(X_test)
#     out = []
#     for i in [6]:
#         for z in [5]:
#             my_tree = Auto_Grow_Tree(X_train,X_valid,y_train,y_valid,X_test,y_test,categories = 1,batch_size = z,num_n = i)
#             node_dict = my_tree.auto_grow()
#             predict_groups = my_tree.final_predict()
#             y_pred_total,y_test_total = [x[0] for x in predict_groups.values()],[x[1] for x in predict_groups.values()]
#             y_pred_final = np.concatenate(y_pred_total)
#             y_test_final = np.concatenate(y_test_total)
#
#
#             # Making the Confusion Matrix
#             from sklearn.metrics import confusion_matrix
#             y_pred_final = (y_pred_final > 0.5)
#             cm = confusion_matrix(y_test_final, y_pred_final)
#             out.append((i,z,cm[0][1]+cm[1][0]))
#     out = sorted(out,key = lambda x : x[2])

if __name__ == '__main__':
    from sklearn import datasets
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    solver = Solver(x_train, x_test, y_train, y_test, './model/test.mod', train=False)
    print(solver.predict())


