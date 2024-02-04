from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np #Optimised version of arrays in python. Used for multidimensional calculations. i.e. working wih matrices
import pandas as pd #Data Analytics tool? Allows us to easily manipulate data. Load, view, edit data sets
import matplotlib as plt #Used to plot graphs, create visual manipulations.
#from IPython.display import clear_output 
#from six.moves import urllib

import tensorflow.compat.v2.feature_column as fc

import tensorflow as ts


#loading data set from googleapis
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')

#Looking at the data sets
print(dftrain.head())
print(dfeval.head())

y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

print(dftrain.head())
print(y_train)
print(y_train["age"])
print(y_train.loc[0])


