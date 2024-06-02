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

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Imports

# %%
import gradio as gr
import os
import pandas as pd
import numpy as np

import time
import matplotlib.pyplot as plt
import plotly.express as px


from notebooks.extraction import extractFeaturesFromFile, extractFeaturesFromFolder, saveFeatures_Comp, saveFeatures_Clustering
from notebooks.extraction import getClustering
from notebooks.extraction import Hyperparams, dictToHyperparams

import zipfile
import shutil

from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

# %% [markdown]
# ## Outline of the necessary functions

# %%
# GLOBALS
feature_dir_cl = "data/features_cluster"
feature_dir_comp = "data/features_comp"
audio_dir_pth = "data/audio_files"
audio_pth = "data/audio_files/20231106_143000.WAV"


# %%
def getClusteringHyperparams(clustering_name):
    hyp_file = os.path.join(feature_dir_cl, f"h_{clustering_name}.csv")
    df_hyp = pd.read_csv(hyp_file)
    return dictToHyperparams(pd.DataFrame.to_dict(df_hyp))


# %%
def runNewClustering(name, zipped_folder, w_dt, w_dt_shift, w_df, w_df_shift, n_fft, n_fft_shift, n_clusters_kmeans, n_pca_components):
    hyp = Hyperparams(w_dt, w_dt_shift, w_df, w_df_shift, n_fft, n_fft_shift, n_clusters_kmeans, n_pca_components)
    t0 = time.time()
    unzipped_folder_path = 'data/unzipped_folder'
    temp_extract_path = 'data/temp_extracted'
    
    # Ensure the unzipped folder path exists
    if not os.path.exists(unzipped_folder_path):
        os.makedirs(unzipped_folder_path)
    
    # Ensure the temp extract path exists
    if not os.path.exists(temp_extract_path):
        os.makedirs(temp_extract_path)
    
    # Clear the unzipped folder path
    for filename in os.listdir(unzipped_folder_path):
        file_path = os.path.join(unzipped_folder_path, filename)
        if os.path.isfile(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    
    # Clear the temp extract path
    for filename in os.listdir(temp_extract_path):
        file_path = os.path.join(temp_extract_path, filename)
        if os.path.isfile(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    
    # Unzip the uploaded folder
    with zipfile.ZipFile(zipped_folder.name, 'r') as zip_ref:
        zip_ref.extractall(temp_extract_path)

    # Move each file from the extracted folder to the unzipped folder path
    for root, dirs, files in os.walk(temp_extract_path):
        for file in files:
            shutil.move(os.path.join(root, file), unzipped_folder_path)
    
    # Remove the temp extracted folder
    shutil.rmtree(temp_extract_path)

    
    df_features, raw_features, _ = extractFeaturesFromFolder(unzipped_folder_path, feature_dir_cl, hyp)
    
    df_pca, mean_pca_values_by_cluster = getClustering(hyp.n_clusters_kmeans, hyp.n_pca_components, df_features, raw_features)
    
    saveFeatures_Clustering(feature_dir_cl, name, df_features, df_pca, mean_pca_values_by_cluster, hyp)


# %%
def listClusterings(feature_dir):
    f_files = [fname[2:-4] for fname in os.listdir(feature_dir) if fname[:2] == "f_"]
    return f_files


# %%
def plotClustering(clustering_name):

    pca_file = os.path.join(feature_dir_cl, f"p_{clustering_name}.csv")
    


# %%
# Function to compare audio file with clusters
def compareAudio(file, clustering_name):
    audio_file_path = file.name

    # Load the clustering
    clustering_file = os.path.join(feature_dir_cl, f"f_{clustering_name}.csv")
    df_features = pd.read_csv(clustering_file)
    
    means_file = os.path.join(feature_dir_cl, f"m_{clustering_name}.csv")
    pca_file = os.path.join(feature_dir_cl, f"p_{clustering_name}.csv")

    # Load the cluster centers and PCA files
    df_means = pd.read_csv(means_file)
    df_pca = pd.read_csv(pca_file)

    hyp = getClusteringHyperparams(clustering_name)
    file_features = extractFeaturesFromFile(audio_file_path, feature_dir_comp, hyp) # Output: Python.dict

    # ========
    # PLOTTING
    # ========

    # Perform PCA on the file features
    pca = PCA(n_components=2)
    pca_file_features = pca.fit_transform(file_features["features"])
    
    df_file_pca = pd.DataFrame(pca_file_features, columns=['PCA1', 'PCA2'])
    df_file_pca['time_windows'] = file_features['time_windows']
    
    # Create Plotly scatter plot for clustered data
    fig = px.scatter(df_pca, x='PCA_1', y='PCA_2', color='Cluster', title='PCA Plot of Audio Features')
    
    # Add black points for the comparison audio file
    fig.add_scatter(x=df_file_pca['PCA1'], y=df_file_pca['PCA2'], mode='markers', marker=dict(color='black'), name='New Audio Features', text=df_file_pca['time_windows'])

    # =====================
    # CLOSEST CLUSTER PLOTS
    # =====================
    
    # Calculate the closest clusters
    pca_columns = [col for col in df_file_pca.columns if col.startswith('PCA')]
    pca_means_columns = [col for col in df_means.columns if col.startswith('PCA')]
    distances = cdist(df_file_pca[pca_columns], df_means[pca_means_columns], metric='euclidean')
    closest_cluster = np.argmin(distances, axis=1)

    df_file_pca['closest_cluster'] = closest_cluster

    # Create a bar plot for the distribution of closest clusters
    cluster_counts = df_file_pca['closest_cluster'].value_counts().reset_index()
    cluster_counts.columns = ['cluster', 'count']
    bar_fig = px.bar(cluster_counts, x='cluster', y='count', title='Distribution of Closest Clusters')
    
    return fig, bar_fig


# %%

# %% [markdown]
# ## GradIO main code

# %%
with gr.Blocks() as demo:
    with gr.Tab("Run the Clustering!"):
        gr.Markdown("### Run the Clustering!")
        name = gr.Textbox(label="Name")

        zipped_folder = gr.File(label="Upload your zipped folder", file_types=['file'])
        
        # Define hyperparameter sliders
        w_dt = gr.Slider(0.25, 1.0, value=0.5, label="w_dt", step=0.25)
        w_dt_shift = gr.Slider(0.25, 1.0, value=0.5, label="w_dt_shift", step=0.25)
        w_df = gr.Slider(2000, 10000, value=5000, label="w_df", step=250)
        w_df_shift = gr.Slider(2000, 10000, value=5000, label="w_df_shift", step=250)
        n_fft = gr.Slider(2, 256, value=128, label="n_fft", step=2)
        n_fft_shift = gr.Slider(2, 256, value=128, label="n_fft_shift", step=2)
        n_clusters_kmeans = gr.Slider(1, 10, value=5, label="n_clusters_kmeans", step=1)
        n_pca_components = gr.Slider(1, 10, value=3, label="n_pca_components", step=1)
        
        run_button = gr.Button("Run!")
        output_text = gr.Textbox(label="Output")

        # Link the button click event to the runNewClustering function
        run_button.click(runNewClustering, inputs=[name, zipped_folder, w_dt, w_dt_shift, w_df, w_df_shift, n_fft, n_fft_shift, n_clusters_kmeans, n_pca_components], outputs=output_text)
    
    with gr.Tab("Compare your Audio..."):
        gr.Markdown("### Compare your Audio...")
        file = gr.File(label="Upload your file", file_types=['audio'])
        clustering_name = gr.Dropdown(choices=listClusterings(feature_dir_cl), label="Clustering")
        #M = gr.Slider(1, 10, label="Number of Closest Clusters (M)")
        compare_button = gr.Button("Compare")

    
    with gr.Tab("Results"):

        cluster_plot = gr.Plot(label="Feature Vectors with Closest Cluster")
        pca_plot = gr.Plot(label="PCA Plot")

    # Link the button click event to the compareAudio function
    compare_button.click(compareAudio, inputs=[file, clustering_name], outputs=[pca_plot, cluster_plot])

demo.launch(share=True)

# %%

# %%
