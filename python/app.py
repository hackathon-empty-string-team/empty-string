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

# %% [markdown]
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
def runNewClustering(name, zipped_folder, w_dt, w_dt_shift, w_df, w_df_shift, n_fft, n_fft_shift, n_clusters_kmeans, n_pca_components, max_files):
    
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

    # Move each file from the extracted folder to the unzipped folder path up to max_files
    count = 0
    for root, dirs, files in os.walk(temp_extract_path):
        for file in files:
            if count < max_files:
                shutil.move(os.path.join(root, file), unzipped_folder_path)
                count += 1
            else:
                break
    
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
def loadFeatures_Comp(feature_file):
    #load the features from the file. Each line is a feature vector, the two last columns are the time windows and name of the file. remove them. Remove also the first column which is the index
    df = pd.read_csv(feature_file)
    features = df.iloc[:, 1:-2].values
    return features


# %%
# Function to compare audio file with clusters
def compareAudio(file, clustering_name, pc_x, pc_y):

    print("PC{}".format(pc_x))
    print("PC{}".format(pc_y))
    audio_file_path = file.name

    # Load the clustering
    clustering_file = os.path.join(feature_dir_cl, f"f_{clustering_name}.csv")
    df_features = pd.read_csv(clustering_file)
    
    means_file = os.path.join(feature_dir_cl, f"m_{clustering_name}.csv")
    pca_file = os.path.join(feature_dir_cl, f"p_{clustering_name}.csv")

    # Load the cluster centers and PCA files
    df_means = pd.read_csv(means_file)
    df_pca = pd.read_csv(pca_file)
    df_pca['Filename'] = df_features['filename']  # Assuming 'filename' column exists
    df_pca['Time Interval'] = df_features['time_window']  # Assuming 'time_windows' column exists
    df_pca['hover_text'] = df_pca.apply(lambda row: f"Filename: {row['Filename']}<br>Time Interval: {row['Time Interval']}", axis=1)

    hyp = getClusteringHyperparams(clustering_name)

    if not os.path.exists(feature_dir_comp):
        os.makedirs(feature_dir_comp)
    file_features = extractFeaturesFromFile(audio_file_path, feature_dir_comp, hyp) # Output: Python.dict

    # ========
    # PLOTTING
    # ========

    # recreate pca from the file
    feature_path = os.path.join(feature_dir_cl, f"f_{clustering_name}.csv")
    features = loadFeatures_Comp(feature_path)

    pca = PCA(n_components=hyp.n_pca_components)
    pca_file_features = pca.fit_transform(file_features["features"])
    
    df_file_pca = pd.DataFrame(pca_file_features, columns=['PCA{}'.format(i) for i in range(hyp.n_pca_components)])
    df_file_pca['time_windows'] = file_features['time_windows']
    df_file_pca['hover_text'] = df_file_pca['time_windows'].apply(lambda x: f"Time Interval: [{x[0]}, {x[1]}]")
    
    # Create Plotly scatter plot for clustered data
    # Create Plotly scatter plot for clustered data
    fig = px.scatter(df_pca, x='PCA_{}'.format(pc_x), y='PCA_{}'.format(pc_y), color='Cluster', title='PCA Plot of Audio Features')
    
    
    # Add black squares for the comparison audio file
    fig.add_scatter(
        x=df_file_pca['PCA{}'.format(pc_x)],
        y=df_file_pca['PCA{}'.format(pc_y)],
        mode='markers',
        marker=dict(color='red', symbol='square', size=6),  # Use 'square' symbol and set size
        name='New',
        text=df_file_pca['hover_text']  # Custom hover text
    )

    fig.update_traces(hovertemplate="<br>".join([
        "Filename: %{customdata[0]}",
        "Time Interval: %{customdata[1]}"
    ]), customdata=df_pca[['Filename', 'Time Interval']].values)


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
    bar_fig.update_layout(
        yaxis_range=[0, cluster_counts['count'].max() + 10],  # Ensure y-axis starts at 0
        xaxis_title='Cluster Number',  # Custom x-axis title
        yaxis_title='Number of Features',  # Custom y-axis title
        width=500,  # Set fixed width
        height=400  # Set fixed height
    )
    
    return fig, bar_fig

# %% [markdown]
# ## GradIO main code

# %%
pcs = [(i + 1) for i in range(10)]

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
        max_files = gr.Slider(0, 50, value=20, label="max_files", step=1)
        
        run_button = gr.Button("Run!")
        output_text = gr.Textbox(label="Output")

        # Link the button click event to the runNewClustering function
        run_button.click(runNewClustering, inputs=[name, zipped_folder, w_dt, w_dt_shift, w_df, w_df_shift, n_fft, n_fft_shift, n_clusters_kmeans, n_pca_components, max_files], outputs=output_text)
    
    with gr.Tab("Compare your Audio..."):
        gr.Markdown("### Compare your Audio...")
        file = gr.File(label="Upload your file", file_types=['audio'])
        if not os.path.exists(feature_dir_cl):
            os.makedirs(feature_dir_cl)
        clustering_name = gr.Dropdown(choices=listClusterings(feature_dir_cl), label="Clustering")
        refresh_button = gr.Button("Refresh")
        compare_button = gr.Button("Compare")

        def refresh_clusterings():
            return gr.Dropdown(choices=listClusterings(feature_dir_cl), interactive=True)
        
        refresh_button.click(refresh_clusterings, outputs=clustering_name)

    
    with gr.Tab("Results"):

        cluster_plot = gr.Plot(label="Feature Vectors with Closest Cluster")
        gr.Markdown("Zoom in to see new features, overlap issues prevail unfortunately...")
        pca_plot = gr.Plot(label="PCA Plot")

        gr.Markdown("### PCA Plot Switcher")
        gr.Markdown("Please only select the PC's as far as you have generated them. Otherwise this will give an error")
    
        # Dropdowns to select principal components
        with gr.Row():
            pc_x = gr.Dropdown(choices=pcs, value=pcs[0], label="Principal Component X")
            pc_y = gr.Dropdown(choices=pcs, value=pcs[1], label="Principal Component Y")
    
        # Button to update the plot
        update_button = gr.Button("Update Plot")
    
        # Update the plot when the button is clicked
        update_button.click(compareAudio, inputs=[file, clustering_name, pc_x, pc_y], outputs=[pca_plot, cluster_plot])

    # Link the button click event to the compareAudio function
    compare_button.click(compareAudio, inputs=[file, clustering_name, pc_x, pc_y], outputs=[pca_plot, cluster_plot])


if __name__ == "__main__":
    demo.launch(share=True)

# %%

# %%

# %%
