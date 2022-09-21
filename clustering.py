import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

csv = 'merged.csv'
k = 2
c = 4


def cleaning_data(csv):
    df = pd.read_csv(csv)
    df_demographic = df.drop(columns=['name', 'birth_date', 'id']).iloc[:, 0:4]
    return df_demographic


def encoding_data(df_demographic):
    df_encoded = pd.get_dummies(df_demographic, columns=['children', 'city'])
    df_scaled = pd.DataFrame(StandardScaler().fit_transform(df_demographic[['age']]), columns=['age'])
    df_preprocessed = pd.concat([df_encoded, df_scaled], axis=1)
    df_preprocessed = df_preprocessed.iloc[:, 1:]
    return df_preprocessed


def pca_generation(df_preprocessed, k):
    pca = PCA(n_components=k)
    H = pca.fit_transform(df_preprocessed)
    H = pd.DataFrame(H)
    return H


def clustering(pca_generation, c):
    clustering = KMeans(n_clusters=c, random_state=42)
    clustering.fit(pca_generation)
    return clustering


def concat_labels(labels, df_demographic):
    df_with_labels = pd.concat([df_demographic, labels], axis=1)
    return df_with_labels


def get_wcs(clustering):
    return clustering.inertia_


def get_labels(clustering):
    return pd.DataFrame(clustering.labels_, columns=['cluster'])


def get_cluster_stats(df_with_labels):
    mean_ = df_with_labels.groupby(['cluster', 'city']).mean()
    return mean_


def main():
    df_demographic = cleaning_data(csv)
    df_preprocessed = encoding_data(df_demographic)
    pca_generation = pca_generation(df_preprocessed, k)
    clustering = clustering(pca_generation, c)
    labels = get_labels(clustering)
    df_with_labels = concat_labels(df_demographic, labels)
    wcs = get_wcs(clustering)
    stats = get_cluster_stats(df_with_labels)


if __name__ == '__main__':
    main()
