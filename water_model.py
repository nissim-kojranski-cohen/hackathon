#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import ensemble, metrics
from sklearn.linear_model import LinearRegression
from joblib import dump, load


# reads file
water = pd.read_csv('water.csv')
water = water.drop(columns=['Unnamed: 0'])
water = water[['name', 'id', 'January', 'February', 'March', 'April',
               'May', 'June', 'July', 'August', 'September', 'October', 'Novemeber',
               'December']]

months = water.columns[2:].values

water = pd.melt(water, 
                    id_vars=['name', 'id'], 
                    value_vars=months, 
                    value_name='price', 
                    var_name='month')

demographics = pd.read_csv('demographics.csv')
demographics = demographics.drop(columns=['Unnamed: 0'])
demographics = demographics[['name', 'birth_date', 'age', 'children', 'employment', 'id', 'city']]

water = pd.merge(water, demographics, how='inner', on='id')

months_dict = {'January':1,
               'February':2,
               'March':3,
               'April':4,
               'May':5,
               'June':6,
               'July':7,
               'August':8,
               'September':9,
               'October':10,
               'Novemeber':11,
               'December':12}


water = water.replace({'month': months_dict})

# dummies
water = pd.get_dummies(water, columns=['city'])
water = water.drop(columns=['name_x', 'name_y', 'id', 'birth_date'])

# ## Split into train and test sets
target = 'price'
X = water.drop(columns=target)
y = water[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# ## Linear Regression Model
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
#
# print('R-Squared on test set:', lr.score(X_test, y_test))
# print('The RMSE on the test set is:', metrics.mean_squared_error(y_test, y_pred) ** 0.5)

# ## Random Forest Regressor

regr = ensemble.RandomForestRegressor()
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)

# print('R-Squared on test set:', regr.score(X_test, y_test))
# print('The RMSE on the test set is:', metrics.mean_squared_error(y_test, y_pred) ** 0.5)

# Since the Random Forest Regressor performs better, we will continue with it
# export model to disk with pickle
dump(regr, 'water_model.pkl')




