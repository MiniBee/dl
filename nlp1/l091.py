import numpy as np
from sklearn.model_selection import train_test_split


def get_data():
    with open('data/ham_data.txt') as ham_f, with open('data/spam_data.ext') as spam_f:
        ham_data = ham_f.readlines()
        spam_data = spam_f.readlines()
        ham_label = np.ones(len(ham_data)).tolist()
        spam_label = np.zeros(len(spam_data)).tolist()
        corpus = ham_data + spam_data
        labels = ham_label + spam_label
    return corpus, labels


def prepare_data(corpus, labels, test_data_proportion=0.3):
    x_train, x_test, y_train, y_test = train_test_split(corpus, labels, test_size=test_data_proportion)
    return x_train, x_test, y_train, y_test





