import gradio as gr
from transformers import pipeline

pipeline = pipeline(task="image-classification", model="julien-c/hotdog-not-hotdog")

def predict(input_img):
    predictions = pipeline(input_img)
    return input_img, {p["label"]: p["score"] for p in predictions}

gradio_app = gr.Interface(
    predict,
    inputs=gr.Image(label="Select hot dog candidate", sources=['upload', 'webcam'], type="pil"),
    outputs=[gr.Image(label="Processed Image"), gr.Label(label="Result", num_top_classes=2)],
    title="Hot Dog? Or Not?",
)

if __name__ == "__main__":
    gradio_app.launch()

# Skeleton of actual frontend here:
"""
import gradio as gr
import os
import pandas as pd
import numpy as np
from datetime import datetime
from feature_extraction_multi_files import process_multiple_audiofiles, process_audio_file
from clustering import perform_clustering
from find_closest_clusters import find_closest_clusters

# Paths
feature_dir = '/python/data/features/'
audio_file_dir = '/python/data/audio_files/'

# Function to save features to CSV
def save_features_to_csv(features_all_files, name, feature_dir):
    all_features = []
    all_time_windows = []
    all_filenames = []
    for file_data in features_all_files:
        filename = file_data["filename"]
        features = file_data["features"]
        time_windows = file_data["time_windows"]
        for i, feature_array in enumerate(features):
            all_features.append(feature_array)
            all_time_windows.append(time_windows[i] if i < len(time_windows) else None)
            all_filenames.append(filename)
    all_features = np.vstack(all_features)
    df_features = pd.DataFrame(all_features, columns=[f'var_{i+1}' for i in range(len(all_features[0]))])
    df_features['time_window'] = all_time_windows
    df_features['filename'] = all_filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(feature_dir, f'{name}_{timestamp}.csv')
    df_features.to_csv(filename)
    return filename

# View 1: Run Clustering
def run_clustering(name, folder, n_pca_comp, n_clusters, stft, n_fft, n_fft_shift, w_dt, w_dt_shift, freq_min, freq_max, w_df, w_df_shift):
    audio_file_dir = folder.name
    features_all_files = process_multiple_audiofiles(audio_file_dir, 5, freq_min, freq_max, w_df, w_df_shift, w_dt, w_dt_shift, n_fft, n_fft_shift)
    feature_file = save_features_to_csv(features_all_files, name, feature_dir)
    df_features = pd.read_csv(feature_file)
    df_pca, cluster_centers = perform_clustering(df_features, n_clusters)
    clustering_filename = os.path.join(feature_dir, f'{name}_cluster_centers_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    pd.DataFrame(cluster_centers).to_csv(clustering_filename, index=False)
    return "Clustering completed and saved under name: " + name

# View 2: Compare Audio
def compare_audio(file, clustering_name):
    audio_file_path, features, time_windows, frequency_windows = process_audio_file(
        file.name, 0, 48000, 4000, 4000, 0.5, 0.5, 512, 256)
    feature_file = os.path.join(feature_dir, f'{clustering_name}.csv')
    df_features = pd.read_csv(feature_file)
    df_pca, cluster_centers = perform_clustering(df_features, 10)
    df_pca = find_closest_clusters(df_pca, cluster_centers)
    return df_pca

# Building Gradio Interface
with gr.Blocks() as demo:
    with gr.Tab("Run Clustering"):
        with gr.Row():
            name = gr.Textbox(label="Name")
            folder = gr.File(label="Upload your folder", file_types=['folder'])
            n_pca_comp = gr.Slider(1, 10, label="n_pca_comp")
            n_clusters = gr.Slider(1, 10, label="n_clusters")
            stft = gr.Checkbox(label="stft")
            n_fft = gr.Slider(256, 1024, label="n_fft")
            n_fft_shift = gr.Slider(256, 1024, label="n_fft_shift")
            w_dt = gr.Slider(0.1, 1.0, label="w_dt")
            w_dt_shift = gr.Slider(0.1, 1.0, label="w_dt_shift")
            freq_min = gr.Number(label="freq_min")
            freq_max = gr.Number(label="freq_max")
            w_df = gr.Slider(1000, 10000, label="w_df")
            w_df_shift = gr.Slider(1000, 10000, label="w_df_shift")
            run_button = gr.Button("Run")
        run_button.click(run_clustering, inputs=[name, folder, n_pca_comp, n_clusters, stft, n_fft, n_fft_shift, w_dt, w_dt_shift, freq_min, freq_max, w_df, w_df_shift], outputs="text")

    with gr.Tab("Compare your Audio"):
        with gr.Row():
            file = gr.File(label="Upload your file", file_types=['audio'])
            clustering_name = gr.Dropdown(choices=["default"], label="Clustering")
            compare_button = gr.Button("Compare")
        compare_button.click(compare_audio, inputs=[file, clustering_name], outputs="dataframe")

demo.launch()

"""