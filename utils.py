# -*- coding: utf-8 -*-
# @Time    : 30/12/2017
# @Author  : Luke

import numpy as np
import pandas as pd


def relu(x):
    return x * (x > 0)


def g_relu(x):
    return 1 * (x > 0)


def softmax(z):
    z -= np.max(z)
    ze = np.exp(z)
    return ze / np.sum(ze, axis=1, keepdims=True)


def g_softmax(z):
    return softmax(z) * (1 - softmax(z))


def cross_entropy(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred))


def load(path, is_test):
    if is_test:
        n_rows = 1024
    else:
        n_rows = None
    data = pd.read_csv(path, header=None, nrows=n_rows)
    labels = data.iloc[:, 0]
    data = data.drop([0], axis=1) / 255.
    return data.values, labels.values


def load_data(is_test=False):
    train_data, train_label = load("mnist/mnist_train.csv", is_test)
    test_data, test_label = load("mnist/mnist_test.csv", is_test)
    return train_data, train_label, test_data, test_label


def parameters_init(input_size, output_size):
    low = - np.sqrt(6. / (input_size + output_size))
    high = -low
    # W = (np.random.rand(input_size,output_size)*2 - 1) * 0.1
    W = np.random.uniform(low, high, (input_size, output_size))
    b = np.zeros((1, output_size))
    return W, b


def ground_truth(labels):
    gt = np.zeros((len(labels), 10))
    for i, value in enumerate(labels):
        gt[i, value] = 1
    return gt
