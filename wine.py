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
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=1)

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

mlp = MLPClassifier(hidden_layer_sizes=(25,25,25), max_iter=500)

mlp.fit(X_train, y_train)
'''
MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=10, learning_rate='constant',
       learning_rate_init=0.001, max_iter=500, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=None,
       shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
       verbose=False, warm_start=False)
'''

# prediction and evaluation
predictions = mlp.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, predictions))
'''
[[  0   0   1   1   0   0]
 [  0   2  21  14   0   0]
 [  0   3 190 168   7   0]
 [  0   0 106 401  37   0]
 [  0   0   5 157  71   0]
 [  0   0   0  23  18   0]]
'''
print(classification_report(y_test, predictions))
'''
                precision    recall  f1-score   support

          3       0.00      0.00      0.00         2
          4       0.40      0.05      0.10        37
          5       0.59      0.52      0.55       368
          6       0.52      0.74      0.61       544
          7       0.53      0.30      0.39       233
          8       0.00      0.00      0.00        41

avg / total       0.52      0.54      0.51      1225
'''