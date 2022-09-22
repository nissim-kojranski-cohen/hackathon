import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from joblib import load
#To Do -> Hot_encode by pikles

pickle = 'clustering_model.pkl'
X_test_file_name = 'X_test_cls.csv'
k = 2

def one_hot_encode_city(df):
    needed_cols = ['children_0', 'children_1','children_2','children_3','children_4','children_5',
                   'city_Herzliya', 'city_Petah_Tikva', 'city_Ramat_Gan','city_Ramat_HaSharon',
                   'city_Rishon_LeTsiyon', 'city_Tel_Aviv_Yaffo',]

    df = pd.get_dummies(df, columns=['city', 'children'])
    existing_cols = df.columns.values
    add_cols = [i for i in needed_cols if i not in existing_cols]
    if len(add_cols) > 0:
        for add_col in add_cols:
            df[add_col] = 0
    return df


def pca_generation(df_preprocessed, k):
    pca = PCA(n_components=k)
    H = pca.fit_transform(df_preprocessed)
    H = pd.DataFrame(H)
    return H


# def encoding_data(X_test):
#     df_encoded = pd.get_dummies(X_test, columns=['children', 'city'])
#     df_scaled = pd.DataFrame(StandardScaler().fit_transform(X_test[['age']]), columns=['age'])
#     df_preprocessed = pd.concat([df_encoded, df_scaled], axis=1)
#     df_preprocessed = df_preprocessed.iloc[:, 1:]
#     return df_preprocessed


model = load(pickle)

X_test = pd.read_csv(X_test_file_name)
df_preprocessed = one_hot_encode_city(X_test)
#X_test = pca_generation(df_preprocessed, k)
prediction = model.predict(X_test)

print(prediction)



