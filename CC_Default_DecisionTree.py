#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 14:57:03 2019

@author: obashaw
"""
# DATA PREP
import pandas as pd
# read data into dataframe
data = pd.read_excel('ccdefault.xls', index_col = 0)

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
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.75)



# MODEL
from sklearn import tree
# Create and fit DT
tree1 = tree.DecisionTreeClassifier(max_depth = 3, min_samples_split = 0.2)
tree1.fit(x_train, y_train)

y_pred = tree1.predict(x_test)

# compute raw accuracy of DT for test data and print
accuracy = 0.0
count = 0
for example in x_test:
    pred = tree1.predict(example.reshape(1, -1))
    if pred == y_test[count]:
        accuracy = accuracy + 1
    count = count + 1

accuracy = accuracy / count
print(accuracy)

# compute AUC
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# VISUALIZATION
# Create DOT data
dot_data = tree.export_graphviz(tree1, out_file=None, feature_names = feature_names[0:23], class_names = ['0', '1'])

# Draw graph
graph = pydotplus.graph_from_dot_data(dot_data)  

# Show graph
graph.write_pdf("cc_default.pdf")