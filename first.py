from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np #Optimised version of arrays in python. Used for multidimensional calculations. i.e. working wih matrices
import pandas as pd #Data Analytics tool? Allows us to easily manipulate data. Load, view, edit data sets
import matplotlib as plt #Used to plot graphs, create visual manipulations.
#from IPython.display import clear_output 
#from six.moves import urllib

import tensorflow.compat.v2.feature_column as fc

import tensorflow as tf


#loading data set from googleapis
dftrain = pd.read_csv('train.csv')
dfeval = pd.read_csv('eval.csv')

#Looking at the data sets
#print(dftrain.head())
#print(dfeval.head())

y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

#print(dftrain.head())
#print(y_train)
#print(dftrain["age"])# index by name
#print(y_train.loc[0])# index by value

#print(dftrain.describe())#gives a little information about the data

#print(dftrain.age.hist(bins=20))


CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']

NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = []
#print(dftrain['deck'].unique())
for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique() #All unique entries 
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

for i in feature_columns:
    print(i)









