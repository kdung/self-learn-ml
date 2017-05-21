# -*- coding: utf-8 -*-
"""
Created on Sun May 21 14:15:31 2017

@author: kibui
"""

import pandas as pd

diabetes = pd.read_csv('Diabetes.csv')
diabetes.head()
diabetes.describe().transpose()
diabetes.shape #(4898, 12)
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

mlp = MLPClassifier(hidden_layer_sizes=(100), max_iter=1000, verbose=True)

mlp.fit(X_train, y_train)
# loss =  0.26141378
train_predictions = mlp.predict(X_train)
train_accuracy = (train_predictions == y_train).sum()/len(y_train)
print('train accuracy %s' % train_accuracy) #0.90625
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

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, predictions))
'''
[[107  16]
 [ 19  50]]
'''
print(classification_report(y_test, predictions))
'''
                precision    recall  f1-score   support

          0       0.85      0.87      0.86       123
          1       0.76      0.72      0.74        69

avg / total       0.82      0.82      0.82       192
'''

len(mlp.coefs_) #4
len(mlp.coefs_[0]) #11
len(mlp.intercepts_[0]) #25