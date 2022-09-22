#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import ensemble, metrics
from sklearn.linear_model import LinearRegression
from joblib import dump, load


# reads file
arnona = pd.read_csv('csvs/arnona.csv')
arnona = arnona.drop(columns=['Unnamed: 0', 'Signed_Up'])

demographics = pd.read_csv('csvs/demographics.csv')
demographics = demographics.drop(columns=['Unnamed: 0'])
demographics = demographics[['name', 'birth_date', 'age', 'children', 'employment', 'id', 'city']]

arnona = pd.merge(arnona, demographics, how='inner', on='id')
arnona = arnona.drop(columns=['name_x', 'name_y', 'id', 'birth_date'])

arnona = pd.get_dummies(arnona, columns=['city'])

# ## Split into train and test sets

target = 'price'
X = arnona.drop(columns=target)
y = arnona[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# ## Linear Regression Model

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

# ## Random Forest Regressor

regr = ensemble.RandomForestRegressor()
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)

# Since the Linear Regression Model performs better, we will continue with it

# export model to disk with pickle
dump(lr, 'pkl/arnona_model.pkl')
