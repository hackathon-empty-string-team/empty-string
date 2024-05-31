# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from scipy.spatial.distance import cdist
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# %%
# config:

feature_dir = '/python/features/'

fname_cluster_centers = feature_dir + 'cluster_centers_kmeans_Nclust10_20240530_130141.csv'

fname_feature_files =  feature_dir + 'sound_features_pca_20240530_130311.csv'

# %%
df_pca_cluster_centers = pd.read_csv(fname_cluster_centers)
df_pca = pd.read_csv(fname_feature_files)

# %%
pca_columns = [col for col in df_pca.columns if col.startswith('PCA')]

distances = cdist(df_pca[pca_columns], df_pca_cluster_centers[pca_columns], metric='euclidean')

closest_clusters = np.argmin(distances, axis=1)

# Add the closest cluster to df_pca_new
df_pca['Closest_Cluster'] = closest_clusters

# %%
cols = ['PCA_1','PCA_2']

plt.scatter(df_pca[cols[0]], df_pca[cols[1]], c=df_pca['Closest_Cluster'],cmap='jet')

plt.xlabel(cols[0])
plt.ylabel(cols[1])
