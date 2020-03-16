#!/usr/bin/env python
# _*_coding:utf-8 _*_
#@Time    :2019/11/24 下午8:56
#@Author  :hongyue pei 
#@FileName: node.py
#@Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt


class Node(object):
    def __init__(self, inbound_notes=[]):
        self.inbound_notes = inbound_notes
        self.value = None
        self.outbound_nodes = []
        self.gradients = {}
        for node in inbound_notes:
            node.outbound_nodes.append(self)

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError


class Input(Node):
    def __init__(self):
        Node.__init__(self)

    def forward(self):
        pass

    def backward(self):
        self.gradients = {self: 0}
        for n in self.outbound_nodes:
            self.gradients[self] += n.gradients[self]


class Linear(Node):
    def __init__(self, X, W, b):
        Node.__init__(self, [X, W, b])

    def forward(self):
        X = self.inbound_notes[0].value
        W = self.inbound_notes[1].value
        b = self.inbound_notes[2].value
        self.value = np.dot(X, W) + b

    def backward(self):
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_notes}
        for n in self.outbound_nodes:
            grad_cost = n.gradients[self]
            self.gradients[self.inbound_notes[0]] += np.dot(grad_cost, self.inbound_notes[1].value.T)
            self.gradients[self.inbound_notes[1]] += np.dot(self.inbound_notes[0].value.T, grad_cost)
            self.gradients[self.inbound_notes[2]] += np.sum(grad_cost, axis=0, keepdims=False)


class Sigmoid(Node):
    def __init__(self, node):
        Node.__init__(self, [node])

    def _sigmoid(self, x):
        return 1. / (1. + np.exp(x))

    def forward(self):
        input_value = self.inbound_notes[0].value
        self.value = self._sigmoid(input_value)

    def backward(self):
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_notes}
        for n in self.outbound_nodes:
            grad_cost = n.gradients[self]
            sigmoid = self.value
            self.gradients[self.inbound_notes[0]] += sigmoid * (1 - sigmoid) * grad_cost


class MSE(Node):
    def __init__(self, y, a):
        Node.__init__(self, [y, a])

    def forward(self):
        y = self.inbound_notes[0].value.reshape(-1, 1)
        a = self.inbound_notes[1].value.reshape(-1, 1)
        self.m = self.inbound_notes[0].value.shape[0]
        self.diff = y - a
        self.value = np.mean(self.diff ** 2)

    def backward(self):
        self.gradients[self.inbound_notes[0]] = (2/self.m) * self.diff
        self.gradients[self.inbound_notes[1]] = (-2 / self.m) * self.diff


# {X: X_, y: y_, W1: W1_0, b1: b1_0, W2: W2_0, b2: b2_0}
def topological_sort(feed_dict):
    input_nodes = [n for n in feed_dict.keys()]
    G = {}
    nodes = [n for n in input_nodes]
    while len(nodes) > 0:
        n = nodes.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outbound_nodes:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodes.append(m)
    L = []
    S = set(input_nodes)
    while len(S) > 0:
        n = S.pop()
        if isinstance(n, Input):
            n.value = feed_dict[n]
        L.append(n)
        for m in n.outbound_nodes:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            if len(G[m]['in']) == 0:
                S.add(m)
    return L


def forward_and_backward(graph):
    for n in graph:
        n.forward()
    for n in graph[::-1]:
        n.backward()


def sgd_update(trainables, learning_rate=1e-2):
    for t in trainables:
        t.value = t.value - learning_rate * t.gradients[t]


from sklearn.utils import resample
from sklearn import datasets


data = datasets.load_iris()
X_ = data.data
y_ = data.target
y_[y_==2] = 1
# print(X_.shape, y_.shape)
np.random.seed(11)
n_features = X_.shape[1]
n_class = 1
n_hidden = 3

X, y = Input(), Input()
W1, b1 = Input(), Input()
W2, b2 = Input(), Input()

l1 = Linear(X, W1, b1)
s1 = Sigmoid(l1)
l2 = Linear(s1, W2, b2)
t1 = Sigmoid(l2)
cost = MSE(y, t1)

W1_0 = np.random.random(X_.shape[1]*n_hidden).reshape([X_.shape[1], n_hidden])
W2_0 = np.random.random(n_hidden * n_class).reshape([n_hidden, n_class])
b1_0 = np.random.random(n_hidden)
b2_0 = np.random.random(n_class)

feed_dict = {X: X_, y: y_, W1: W1_0, b1: b1_0, W2: W2_0, b2: b2_0}

epochs = 1000
m = X_.shape[0]
batch_size = 20
steps_per_epoch = m // batch_size
lr = 0.1

graph = topological_sort(feed_dict)
trainables = [W1, b1, W2, b2]

l_Mat_W1 = [W1_0]
l_Mat_W2 = [W2_0]

l_loss = []
for i in range(epochs):
    loss = 0
    for j in range(steps_per_epoch):
        X_batch, y_batch =resample(X_, y_, n_samples=batch_size)
        X.value = X_batch
        y.value = y_batch

        forward_and_backward(graph)
        sgd_update(trainables, lr)
        loss += graph[-1].value
    l_loss.append(loss)

    if i % 10 == 9:
        print('Eporch %d, Loss = %1.5f' % (i, loss))










