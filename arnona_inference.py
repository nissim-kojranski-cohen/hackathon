from flask import Flask, request
import pickle
import pandas as pd
import numpy as np
from joblib import dump, load

ordered_cols_test = ['age', 'children', 'employment', 'city_Herzliya',
       'city_Petah_Tikva', 'city_Ramat_Gan', 'city_Ramat_HaSharon',
       'city_Rishon_LeTsiyon', 'city_Tel_Aviv_Yaffo']

pickle_name = 'arnona_model.pkl'
X_test_file_name = 'X_test.csv'


def one_hot_encode_city(df):
    needed_cols = ['city_Herzliya', 'city_Petah_Tikva', 'city_Ramat_Gan',
                   'city_Ramat_HaSharon', 'city_Rishon_LeTsiyon', 'city_Tel_Aviv_Yaffo']
    df = pd.get_dummies(df, columns=['city'])
    existing_cols = df.columns.values
    add_cols = [i for i in needed_cols if i not in existing_cols]
    if len(add_cols) > 0:
        for add_col in add_cols:
            df[add_col] = 0
    return df

# with open(pickle_name, 'rb') as input_model:
#     model = pickle.load(input_model)

model = load(pickle_name)

X_test = pd.read_csv(X_test_file_name)
X_test = one_hot_encode_city(X_test) # one hot encode the city
X_test = X_test[ordered_cols_test] # order columns according to fit

y_pred = model.predict(X_test)
print(y_pred)