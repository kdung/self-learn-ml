#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 18:12:10 2017

@author: sophie
"""

# PNN

import pandas as pd

wine = pd.read_csv('winequality-white.csv')
wine.head()
wine.describe().transpose()
wine.shape #(4898, 12)
X = wine.drop('quality', axis=1)
y = wine['quality']


from neupy.algorithms import PNN

# train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1)

# data preprocessing
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# fit only to the training data
scaler.fit(X_train)
#StandardScaler(copy=True, with_mean=True, with_std=True)

# apply the transformations to the data
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


pnn_network = PNN(std=1, shuffle_data= True, batch_size=10, step=0.01, verbose=False)
pnn_network.train(X_train, y_train)
result = pnn_network.predict(X_test)

train_predictions = pnn_network.predict(X_train)
train_accuracy = (train_predictions == y_train).sum()/len(y_train)
print('train accuracy %s' % train_accuracy) #0.958907605921

# prediction and evaluation
predictions = pnn_network.predict(X_test)

test_accuracy = (predictions == y_test).sum()/len(y_test)
print('test accuracy %s' % test_accuracy) #0.617346938776
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, predictions))
'''
[[  0   0   2   0   0   0]
 [  0   4  18   4   2   1]
 [  0   8 181  81  18   5]
 [  0   7  61 308  49  10]
 [  0   2  15  50 110   7]
 [  0   0   2   5  13  17]]
'''
print(classification_report(y_test, predictions))
