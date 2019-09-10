#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 10:54:31 2019

@author: obashaw
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
def svr_param_selection(X, y):
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [1e-7, 1e-6, 1e-5, 1e-4, 0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(svm.SVR(kernel='rbf'), param_grid)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_

def svr_fit(X, y):
    # create and fit SVM model using best parameters
    clf = svm.SVR(gamma = 'scale', C = 1, kernel = 'rbf')
    clf.fit(X, y)
    return clf

# DATA PREP
import pandas as pd
import numpy as np
# read data into dataframe
data = pd.read_excel('default_of_credit_card_clients.xls', index_col = 0)

# set column headers to first row (descriptions) and drop the 'ID' row
data.columns = data.iloc[0]
data = data.drop('ID')

# get list of feature names for export_graphviz
feature_names = data.columns.tolist()

# separate our 1D label array from 2D attributes array and set to float64
y = data.pop('default payment next month').values
y = y.astype('float64')

# take dataframe to numpy 2D arraay
X = data.values

# split training and testing X,y groups
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)


#MODEL
from sklearn import svm
import pickle
best = svr_param_selection(x_train, y_train)

#clf = svr_fit(x_train, y_train)
#pickle.dump(clf, open("CC_Default_SVR.pickle", "wb"))
clf = pickle.load(open("CC_Default_SVR.pickle", "rb"))
clf.score(x_train, y_train)

y_pred = np.array([])
for x in x_test:
    y_pred = np.append(y_pred, clf.predict(x.reshape(1, -1)))

for i in range(0, 9000):
    print("Real: ", y_test[i], " Pred: ", y_pred[i])






