# -*- coding: utf-8 -*-
"""
Created on Sun May 21 12:08:42 2017

@author: kibui
"""

import pandas as pd

wine = pd.read_csv('winequality-white.csv')
wine.head()
wine.describe().transpose()
wine.shape #(4898, 12)
X = wine.drop('quality', axis=1)
y = wine['quality']

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

# training the model
# using MLP classifier model

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(100), max_iter=500)

mlp.fit(X_train, y_train)

train_predictions = mlp.predict(X_train)
train_accuracy = (train_predictions == y_train).sum()/len(y_train)
print('train accuracy %s' % train_accuracy) #0.650149741356
'''
MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=100, learning_rate='constant',
       learning_rate_init=0.001, max_iter=500, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=None,
       shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
       verbose=True, warm_start=False)
'''

# plot learning curve
from sklearn.model_selection import ShuffleSplit
from plot_learning_curve import plot_learning_curve
title = "Learning Curves"
estimator = MLPClassifier(hidden_layer_sizes=(100), max_iter=1000)
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
plt = plot_learning_curve(estimator, title, X_train, y_train, (0.0, 1.01), cv=cv, n_jobs=1)
plt.show()

# prediction and evaluation
predictions = mlp.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, predictions))
'''
[[  0   0   1   1   0   0]
 [  0   6  21  10   0   0]
 [  1   3 224 134   5   1]
 [  0   5 119 365  50   5]
 [  0   0  10 137  78   8]
 [  0   0   1  21  16   3]]
'''
print(classification_report(y_test, predictions))
'''
                precision    recall  f1-score   support

          3       0.00      0.00      0.00         2
          4       0.43      0.16      0.24        37
          5       0.60      0.61      0.60       368
          6       0.55      0.67      0.60       544
          7       0.52      0.33      0.41       233
          8       0.18      0.07      0.10        41

avg / total       0.54      0.55      0.54      1225
'''

len(mlp.coefs_) #4
len(mlp.coefs_[0]) #11
len(mlp.intercepts_[0]) #25