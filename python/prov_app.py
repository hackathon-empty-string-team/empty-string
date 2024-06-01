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

# %% [markdown]
# # Provisional GradIO app notebook

# %% [markdown]
# The other file is a generated skeleton

# %%
import gradio as gr
import os
import pandas as pd
import numpy as np
from datetime import datetime
from notebooks.extraction import extractFeaturesFromFile, extractFeaturesFromFolder, getClustering, saveFeatures_Comp, saveFeatures_Clustering 

# %%
# time sampling parameters
w_dt = 0.5 # time window of each sample [sec]
w_dt_shift = 0.5 # time by which samples are shifted [sec]

# frequency sampling parameters
w_df = 4000 # ferquency indow of each sample [Hz]
w_df_shift = 4000 # ferquency by which windows are shifted [Hz]

# fft parameters
n_fft = 512 # number of sampling points in sub-samples used for fft (sets time resolution of spectra)
n_fft_shift = 256 # number of sampling points by which fft sub-samples are shifted

freq_min, freq_max =  0.0, 48000.0 # min/max frequency of spectra [Hz]

n_clusters_kmeans = 10
n_pca_components = 8

# Two feature directories, one for the clustering features and one for the comparison features
feature_dir_cl = "../data/features_cluster"
feature_dir_comp = "../data/features_comp"
audio_dir_pth = "../data/audio_files"
audio_pth = "../data/audio_files/20231106_143000.WAV"

name = "my-cluster-1"

t0 = time.time()

single_audio_features = extractFeaturesFromFile(audio_pth, feature_dir_comp, w_dt, w_dt_shift, w_df, w_df_shift, n_fft, n_fft_shift, freq_min, freq_max, n_clusters_kmeans, n_pca_components)
df_features, raw_features, correlation_matrix = extractFeaturesFromFolder(audio_dir_pth, feature_dir_cl, w_dt, w_dt_shift, w_df, w_df_shift, n_fft, n_fft_shift, freq_min, freq_max, n_clusters_kmeans, n_pca_components)
t1 = time.time() - t0

# %%
df_pca, mean_pca_values_by_cluster = getClustering(n_clusters_kmeans, n_pca_components, raw_features)

# %%
saveFeatures_Comp(feature_dir_comp, single_audio_features)

# %%
saveFeatures_Clustering(feature_dir_cl, name, df_features, df_pca, mean_pca_values_by_cluster)
