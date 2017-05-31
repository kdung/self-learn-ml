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

#mlp = MLPClassifier(hidden_layer_sizes=(100), max_iter=500)
mlp = MLPClassifier(activation='tanh', learning_rate_init=1,
                    learning_rate="adaptive", solver='sgd',
                    verbose=False, hidden_layer_sizes=(200), max_iter=1000)

mlp.fit(X_train, y_train)

train_predictions = mlp.predict(X_train)
train_accuracy = (train_predictions == y_train).sum()/len(y_train)
print('train accuracy %s' % train_accuracy) #0.958907605921
'''
MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=100, learning_rate='constant',
       learning_rate_init=0.001, max_iter=500, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=None,
       shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
       verbose=True, warm_start=False)
'''
# loss = 0.38452713

# prediction and evaluation
predictions = mlp.predict(X_test)

test_accuracy = (predictions == y_test).sum()/len(y_test)
print('test accuracy %s' % test_accuracy) #0.617346938776
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, predictions))
'''
[[  0   0   2   0   0   0]
 [  0   7  18   2   2   0]
 [  1  10 183  89   9   1]
 [  0   5  67 313  47   3]
 [  0   0   9  74  94   7]
 [  0   0   2  11  14  10]]
'''
print(classification_report(y_test, predictions))
'''
                precision    recall  f1-score   support

          3       0.00      0.00      0.00         2
          4       0.32      0.24      0.27        29
          5       0.65      0.62      0.64       293
          6       0.64      0.72      0.68       435
          7       0.57      0.51      0.54       184
          8       0.48      0.27      0.34        37

avg / total       0.61      0.62      0.61       980
'''

len(mlp.coefs_) #4
len(mlp.coefs_[0]) #11
len(mlp.intercepts_[0]) #25

'''
# plot learning curve
from sklearn.model_selection import ShuffleSplit
from plot_learning_curve import plot_learning_curve
title = "Learning Curves"

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
plt = plot_learning_curve(mlp, title, X_train, y_train, (0.0, 1.01), cv=cv, n_jobs=1)
plt.show()
'''
