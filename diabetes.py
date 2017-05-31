# -*- coding: utf-8 -*-
"""
Created on Sun May 21 14:15:31 2017

@author: kibui
"""

import pandas as pd

diabetes = pd.read_csv('Diabetes.csv')
diabetes.head()
diabetes.describe().transpose()
diabetes.shape
X = diabetes.drop('Classify', axis=1)
y = diabetes['Classify']

# train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

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

mlp = MLPClassifier(hidden_layer_sizes=(8), max_iter=1000, verbose=False)

mlp.fit(X_train, y_train)
# loss = 0.23052485
train_predictions = mlp.predict(X_train)
train_accuracy = (train_predictions == y_train).sum()/len(y_train)
print('train accuracy %s' % train_accuracy) #0.93055555555555558 0.973958333333
'''
MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=100, learning_rate='constant',
       learning_rate_init=0.001, max_iter=1000, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=None,
       shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
       verbose=True, warm_start=False)
'''

# prediction and evaluation
predictions = mlp.predict(X_test)
test_accuracy = (predictions == y_test).sum()/len(y_test)
print('test accuracy %s' % test_accuracy) #0.755208333333
'''
train accuracy 0.765625
test accuracy 0.807291666667
'''
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, predictions))
'''
[[107  16]
 [ 21  48]]
'''
print(classification_report(y_test, predictions))
'''
                precision    recall  f1-score   support

          0       0.84      0.87      0.85       123
          1       0.75      0.70      0.72        69

avg / total       0.81      0.81      0.81       192
'''

len(mlp.coefs_) #4
len(mlp.coefs_[0]) #11
len(mlp.intercepts_[0]) #25

# plot learning curve
"""
from sklearn.model_selection import ShuffleSplit
from plot_learning_curve import plot_learning_curve
title = "Learning Curves"
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
plt = plot_learning_curve(mlp, title, X_train, y_train, (0.0, 1.01), cv=cv, n_jobs=1)
plt.show()
"""