#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 18:38:47 2017

@author: sophie
"""

from operator import itemgetter

import numpy as np
from sklearn import datasets, grid_search
from sklearn.model_selection import train_test_split
from neupy import algorithms, estimators, environment


environment.reproducible()


def scorer(network, X, y):
    result = network.predict(X)
    return estimators.rmsle(result, y)


import pandas as pd

wine = pd.read_csv('winequality-white.csv')
wine.head()
wine.describe().transpose()
wine.shape #(4898, 12)
X = wine.drop('quality', axis=1)
y = wine['quality']
y = np.array(y)
X = np.array(X)


x_train, x_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7
)

dataset = datasets.load_diabetes()
x_train, x_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, train_size=0.7
)

grnnet = algorithms.GRNN(std=0.5, verbose=True)
grnnet.train(x_train, y_train)
error = scorer(grnnet, x_test, y_test)
print("GRNN RMSLE = {:.3f}\n".format(error))

