import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from joblib import load

cls_pickle = './pkl/clustering_model.pkl'
pca_pickle = './pkl/pca_model.pkl'
k = 2

def one_hot_encode_city(df):
    needed_cols = ['city_Herzliya', 'city_Petah_Tikva', 'city_Ramat_Gan', 'city_Ramat_HaSharon',
                   'city_Rishon_LeTsiyon', 'city_Tel_Aviv_Yaffo', ]

    df = pd.get_dummies(df, columns=['city'])
    existing_cols = df.columns.values
    add_cols = [i for i in needed_cols if i not in existing_cols]
    if len(add_cols) > 0:
        for add_col in add_cols:
            df[add_col] = 0
    return df


model = load(cls_pickle)
pca = load(pca_pickle)

df_preprocessed = one_hot_encode_city(X_test)

df_preprocessed = df_preprocessed[['children', 'employment', 'city_Herzliya', 'city_Petah_Tikva',
                                   'city_Ramat_Gan', 'city_Ramat_HaSharon', 'city_Rishon_LeTsiyon',
                                   'city_Tel_Aviv_Yaffo', 'age']]

pca = pca.transform(df_preprocessed)

prediction = model.predict(pca)

print(prediction)
