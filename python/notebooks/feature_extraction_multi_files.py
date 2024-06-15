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
# # Imports

# %%
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MeanShift
import plotly.express as px
from datetime import datetime
import shutil
from collections import defaultdict
import random


# %% [markdown]
# # Helper Functions

# %%
def get_windows(val_min, val_max, window_width, window_shift, upper_bound=40000):
    # Input validation
    if not isinstance(val_min, (int, float)) or not isinstance(val_max, (int, float)):
        raise ValueError("val_min and val_max must be int or float")
    if not isinstance(window_width, (int, float)) or not isinstance(window_shift, (int, float)):
        raise ValueError("window_width and window_shift must be int or float")
    if val_min >= val_max:
        raise ValueError("val_min must be less than val_max")
    if window_width <= 0 or window_shift <= 0:
        raise ValueError("window_width and window_shift must be positive values")
    val_max = min(upper_bound, val_max)
    windows = []
    current_start = val_min
    # Calculate the number of bins
    n_bin = int((val_max - val_min) / window_shift)
    
    for i in range(n_bin):
        lo = current_start
        hi = lo + window_width

        # Ensure the upper bound does not exceed val_max
        if hi > val_max:
            break

        windows.append([lo, hi])
        current_start += window_shift

    return windows

def plot_spectrogram_with_windows(spec_db, sr, time_windows, frequency_windows, n_fft, hop_length):
    
    librosa.display.specshow(spec_db, sr=sr, x_axis='time', y_axis='linear', cmap='rainbow', n_fft=n_fft, hop_length=hop_length)
    
    for lo, hi in time_windows:
        plt.axvline(x=lo, ls='-', c='r', lw=0.5)
        plt.axvline(x=hi, ls='-', c='r', lw=0.5)
    
    for i, (lo, hi) in enumerate(frequency_windows):
        plt.axhline(y=lo,ls='-',c='r', lw=0.5)
        plt.axhline(y=hi,ls='-',c='r', lw=0.5)

    plt.show()

def apply_conv2d(S_db_mod, n_filters, kernel_size, padding):
    """
    Applies a Conv2D layer followed by global average pooling to a modified spectrogram.

    NOTE: you need to set the manual_seed, otherwise the weight initiation of the layers 
    is different for every file, resulting in random feature extraction and it does not work.

    Parameters:
    S_db_mod (numpy.ndarray): Modified spectrogram in dB.

    Returns:
    numpy.ndarray: Output vector after Conv2D and global average pooling.
    """
    torch.manual_seed(42)

    # Convert the numpy array to a torch tensor and add batch and channel dimensions
    input_tensor = torch.tensor(S_db_mod, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    # Define a Conv2D layer with 256 filters, 10x10 kernel size, and 'same' padding
    conv_layer = nn.Conv2d(
        in_channels=1,
        out_channels=n_filters,
        kernel_size=kernel_size,
        padding=padding,
        bias=True
    )
    
    # Initialize weights using GlorotUniform (Xavier) initialization
    nn.init.xavier_uniform_(conv_layer.weight, gain=nn.init.calculate_gain('relu'))
    
    # Apply the Conv2D layer
    output_tensor = conv_layer(input_tensor)
    
    # Apply global average pooling
    gap_layer = nn.AdaptiveAvgPool2d((1, 1))
    output_vector = gap_layer(output_tensor).squeeze()
    
    # Convert the output vector to a numpy array
    output_vector_np = output_vector.detach().numpy()

    # Convert the output vector to a numpy array and add the batch dimension
    output_vector_np = output_vector.squeeze().unsqueeze(0).detach().numpy()
    
    return output_vector_np

def apply_conv2d_double_whopper(S_db_mod, n_filters, kernel_size, padding):
    """
    Applies two Conv2D layers followed by global average pooling to a modified spectrogram.

    NOTE: you need to set the manual_seed, otherwise the weight initiation of the layers 
    is different for every file, resulting in random feature extraction and it does not work.

    Parameters:
    S_db_mod (numpy.ndarray): Modified spectrogram in dB.
    n_filters (int): Number of filters for the convolutional layers.
    kernel_size (int or tuple): Size of the convolutional kernels.
    padding (int or tuple): Padding for the convolutional layers.

    Returns:
    numpy.ndarray: Output vector after Conv2D and global average pooling.
    """
    torch.manual_seed(42)

    # Convert the numpy array to a torch tensor and add batch and channel dimensions
    input_tensor = torch.tensor(S_db_mod, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    # Define the first Conv2D layer
    conv_layer1 = nn.Conv2d(
        in_channels=1,
        out_channels=n_filters,
        kernel_size=kernel_size,
        padding=padding,
        bias=True
    )
    
    # Initialize weights using GlorotUniform (Xavier) initialization
    nn.init.xavier_uniform_(conv_layer1.weight, gain=nn.init.calculate_gain('relu'))
    
    # Define the second Conv2D layer
    conv_layer2 = nn.Conv2d(
        in_channels=n_filters,
        out_channels=n_filters * 2,  # Optionally double the number of filters
        kernel_size=kernel_size,
        padding=padding,
        bias=True
    )
    
    # Initialize weights using GlorotUniform (Xavier) initialization
    nn.init.xavier_uniform_(conv_layer2.weight, gain=nn.init.calculate_gain('relu'))
    
    # Apply the first Conv2D layer followed by ReLU activation
    output_tensor = conv_layer1(input_tensor)
    output_tensor = nn.ReLU()(output_tensor)
    
    # Apply the second Conv2D layer followed by ReLU activation
    output_tensor = conv_layer2(output_tensor)
    output_tensor = nn.ReLU()(output_tensor)
    
    # Apply global average pooling
    gap_layer = nn.AdaptiveAvgPool2d((1, 1))
    output_vector = gap_layer(output_tensor).squeeze()
    
    # Convert the output vector to a numpy array
    output_vector_np = output_vector.detach().numpy()
    
    return output_vector_np
    
def get_features(spec_db, times, time_windows, frequencies, frequency_windows, n_filters, kernel_size, padding):
    features = []

    for i, (tlo, thi) in enumerate(time_windows):
    
        features_in_time_frequency_window = []
        
        for j, (flo, fhi) in enumerate(frequency_windows):

            assert tlo != thi, "Tlo is equal thi"
            
            t_mask = (times >= tlo) & (times <= thi)
            f_mask = (frequencies >= flo) & (frequencies <= fhi)

            #print(flo, fhi, frequencies)
        
            # Extract the subsample based on the masks
            subsample = spec_db[f_mask][:, t_mask]

            # normalize the subsample with the absolute maximum value
            subsample = subsample / np.max(np.abs(subsample))
       
            output_vector_np = apply_conv2d_double_whopper(subsample, n_filters, kernel_size, padding)            

            # print('subsample shape ', subsample.shape)
            # print('feature vector shape ', output_vector_np.shape)
                    
            features_in_time_frequency_window.append(output_vector_np.flatten())
    
        features.append(np.array(features_in_time_frequency_window))

    return features



def process_audio_file(audio_file_path, freq_min, freq_max, w_df, w_df_shift, w_dt, w_dt_shift, n_fft, n_fft_shift, n_filters, kernel_size, padding):
    """
    Process an audio file to extract features based on specified time and frequency windows.
    
    Parameters:
    - audio_file_path: Path to the audio file.
    - freq_min: Minimum frequency for frequency windows.
    - freq_max: Maximum frequency for frequency windows.
    - w_df: Width of each frequency window.
    - w_df_shift: Shift between successive frequency windows.
    - w_dt: Width of each time window in seconds.
    - w_dt_shift: Shift between successive time windows in seconds.
    - n_fft: The FFT window size.
    - n_fft_shift: The number of samples between successive frames.
    - get_windows: Function to generate windows.
    - get_features: Function to extract features from the spectrogram.
    
    Returns:
    - features: File path, Extracted features, time windows and frequency windows
    """
    # Read the audio file
    amp, sr = librosa.load(audio_file_path, sr=None)  # Load amplitude and sampling rate from file

    # Get spectrogram
    spec = librosa.stft(amp, n_fft=n_fft, hop_length=n_fft_shift)
    spec_db = librosa.amplitude_to_db(np.abs(spec))

    # Set time windows
    n_sec_in_audio_file = len(amp) / sr
    time_windows = get_windows(0, n_sec_in_audio_file, w_dt, w_dt_shift)
    
    # plot_spectrogram_with_windows(spec_db, sr, time_windows, frequency_windows, n_fft, n_fft_shift)
    
    times = librosa.frames_to_time(np.arange(spec_db.shape[1]), sr=sr, n_fft=n_fft, hop_length=n_fft_shift)

    frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # Set frequency windows
    frequency_windows = get_windows(freq_min, freq_max, w_df, w_df_shift, upper_bound=frequencies[-1])

    # Feature extraction
    features = get_features(spec_db, times, time_windows, frequencies, frequency_windows, n_filters, kernel_size, padding)
    
    
    return audio_file_path, features, time_windows, frequency_windows

def process_multiple_audiofiles(audio_file_dir, max_files, freq_min, freq_max, w_df, w_df_shift, w_dt, w_dt_shift, n_fft, n_fft_shift, n_filters, kernel_size, padding):
    features_all_files = []

    # Counter to limit the number of processed files
    file_count = 0

    # Loop over filenames in the directory
    for filename in os.listdir(audio_file_dir):
        
        if (filename.lower().endswith(".wav") or filename.lower().endswith(".mp3")) and file_count < max_files:  # Process only .wav files and limit to max_files
            file_path = os.path.join(audio_file_dir, filename)
            #('processing ', filename)

            start_time = time.time()

            audio_file_path, features, time_windows, frequency_windows = process_audio_file(
                file_path,
                freq_min, freq_max,
                w_df, w_df_shift,
                w_dt, w_dt_shift,
                n_fft, n_fft_shift,
                n_filters, kernel_size, padding
            )

            # Create a dictionary for the current file
            file_features = {
                "filename": filename,
                "audio_file_path": audio_file_path,
                "features": features,
                "time_windows": time_windows,
                "frequency_windows": frequency_windows
            }
            
            # Append the dictionary to the list
            features_all_files.append(file_features)

            execution_time = time.time() - start_time
            print(f"Execution time: {execution_time:.2f} seconds \n")

            # Increment the file counter
            file_count += 1

        # Break the loop if we have processed the desired number of files
        if file_count >= max_files:
            break

    # Print a message after processing the limited number of files
    print(f"Processed {file_count} files.")
    
    return features_all_files

def count_none_frequency_windows(features_all_files):
    none_counts = []

    for file_features in features_all_files:
        if file_features.get("frequency_windows") is None:
            none_counts.append(file_features["filename"])

    # Return the count and filenames with None frequency_windows
    return len(none_counts), none_counts


def plot_cluster_vs_absolute_time(df_pca):
    filenames = df_pca["filename"].to_numpy()
    time_windows = df_pca["time_window"].to_numpy()
    clusters = df_pca["Cluster"].to_numpy()

    absolute_times = []
    for filename, time_window in zip(filenames, time_windows):
        time_ymd = filename.split('_')[1]
        time_ymd = time_ymd.split('.')[0]
        time_hms = filename.split('_')[2]
        time_hms = time_hms.split('.')[0]
        # transform time_str to absolute time in seconds
        time_str = f'{time_ymd}{time_hms}'
        time_sec = 0
        time_sec = int(time_str[-2:])  # seconds
        time_sec += time_window[0]  # beginning of time window
        time_str = time_str[:-2]
        time_sec += int(time_str[-2:]) * 60  # minutes
        time_str = time_str[:-2]
        time_sec += int(time_str[-2:]) * 60 * 60  # hours
        time_str = time_str[:-2]
        time_sec += int(time_str) * 60 * 60 * 24  # days
        time_str = time_str[:-2]
        time_sec += int(time_str) * 60 * 60 * 24 * 30  # months
        time_str = time_str[:-2]
        time_sec += int(time_str) * 60 * 60 * 24 * 30 * 365  # years
        absolute_times.append(time_sec)

    absolute_times = np.array(absolute_times)

    # substraction of the first time to have time in seconds from the beginning of the recording
    absolute_times -= np.min(absolute_times)

    # transform absolute time to hours
    absolute_times /= 3600

    # plot cluster vs absolute time
    plt.figure()
    plt.scatter(absolute_times, clusters)
    plt.xlabel('Absolute time [hours]')
    plt.ylabel('Cluster')
    plt.title('Cluster vs Absolute time')
    plt.show()
    
    #plot each cluster histogram, each in a subplot, keep the same x-axis
    for cluster in np.unique(clusters):
        plt.figure()
        plt.hist(absolute_times[clusters == cluster], range=(0, np.max(absolute_times)), bins=20)
        plt.xlabel('Absolute time [hours]')
        plt.ylabel('Frequency')
        plt.title(f'Cluster {cluster} vs Absolute time')
        plt.show()


# %% [markdown]
# # GradIO functions & testing

# %% [markdown]
# ## feature extraction & clustering functions

# %%
def extractFeaturesFromFile(audio_pth, feature_dir, w_dt, w_dt_shift, w_df, w_df_shift, n_fft, n_fft_shift, freq_min, freq_max, n_clusters_kmeans, n_pca_components, n_filters, kernel_size, padding):

    audio_file_path, features, time_windows, frequency_windows = process_audio_file(
                audio_pth,
                freq_min, freq_max,
                w_df, w_df_shift,
                w_dt, w_dt_shift,
                n_fft, n_fft_shift,
                n_filters, kernel_size, padding
            )

    # Create a dictionary for the current file
    file_features = {
        "filename": audio_pth.split('/')[-1],
        "audio_file_path": audio_file_path,
        "features": features,
        "time_windows": time_windows,
        "frequency_windows": frequency_windows
    }
    # make sure there is no invalid window processing going on 

    assert file_features.get("frequency_windows") is not None, "Files with None frequency windows"

    return file_features


# %%
def extractFeaturesFromFolder(audio_dir_pth, feature_dir, max_files, w_dt, w_dt_shift, w_df, w_df_shift, n_fft, n_fft_shift, freq_min, freq_max, n_filters, kernel_size, padding):

    features_all_files = process_multiple_audiofiles(audio_file_dir, max_files, freq_min, freq_max, w_df, w_df_shift, w_dt, w_dt_shift, n_fft, n_fft_shift, n_filters, kernel_size, padding)

    # construct dataframe with features from all files + metadata
    df_features_all_files = []
    
    for file_id, file_data in enumerate(features_all_files):
        filename = file_data["filename"]
        features = file_data["features"]
        frequency_windows = file_data["frequency_windows"]
        time_windows = file_data["time_windows"]
        
        # Create an empty list to collect DataFrames for each time window
        df_features_in_file = []
        
        # Loop over time windows
        for time_window_id, (features_in_time_window, time_window) in enumerate(zip(features, time_windows)):
    
            # Construct DataFrame with one row per frequency window
            df_features_in_time_window = pd.DataFrame(features_in_time_window, columns=[f'var_{j+1}' for j in range(len(features_in_time_window[0]))])
    
            df_features_in_time_window['frequency_window'] = [list(row) for row in frequency_windows]
            df_features_in_time_window['time_window'] = [list(time_window)] * len(frequency_windows)
    
            # Add columns for time window ID and frequency window ID
            df_features_in_time_window['time_window_id'] = time_window_id
            df_features_in_time_window['frequency_window_id'] = range(len(frequency_windows))

            # Append the DataFrame to the list
            df_features_in_file.append(df_features_in_time_window)

        
        # Concatenate all DataFrames in the list to create a single DataFrame
        df_features_in_file = pd.concat(df_features_in_file, ignore_index=True)

        df_features_in_file['file_id'] = file_id
        df_features_in_file['filename'] = filename
    
        df_features_all_files.append(df_features_in_file)
    
    # Concatenate all DataFrames in the list to create a single DataFrame
    df_features_all_files = pd.concat(df_features_all_files, ignore_index=True)
    
    # Identify the features with the highest values
    # Here we assume "highest values" means the highest mean value for simplicity
    # Exclude non-numeric columns
    numeric_df_features = df_features_all_files.select_dtypes(include=[np.number])
    
    # Calculate the mean values of each column (feature)
    mean_values = numeric_df_features.mean()
    
    # Identify the top 100 features with the highest mean values
    top_features = mean_values.nlargest(100).index
    
    # Keep only these columns in the DataFrame
    df_features_reduced = numeric_df_features[top_features]
    
    # Calculate the correlation matrix
    correlation_matrix = df_features_reduced.corr()

    return df_features_all_files, correlation_matrix


# %%
def get_pca_clustering(n_clusters_kmeans, n_pca_components, df_features):

    feature_columns = [col for col in df_features.columns if col.startswith('var')]
    pca_columns = [f'PCA_{i+1}' for i in range(n_pca_components)]
    
    df_pca = []
    df_cluster_centers = []

    # loop over frequency windows
    for i in df_features.frequency_window_id.unique():

        print('frequency window', i)

        # get raw feature vector
        select_frequency_window = df_features.frequency_window_id==i
    
        df_features_tmp = df_features[select_frequency_window].copy()
    
        raw_features_tmp = np.array(
            df_features_tmp[feature_columns]
        )
        
        # get pca components for each feature vector
        pca = PCA(n_components=n_pca_components)
        
        features_pca_tpm = pca.fit_transform(
            raw_features_tmp
        )
    
        # KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters_kmeans, random_state=0).fit(
            features_pca_tpm
        )
    
        # make pca dataframe
        df_pca_tmp = pd.DataFrame(
            features_pca_tpm, columns=pca_columns
        )
    
        # Add the time windows, frequency windows, filenames, and cluster labels to the DataFrame
        df_pca_tmp['frequency_window'] = df_features_tmp['frequency_window'].values
        df_pca_tmp['time_window'] = df_features_tmp['time_window'].values
        df_pca_tmp['frequency_window_id'] = df_features_tmp['frequency_window_id'].values
        df_pca_tmp['time_window_id'] = df_features_tmp['time_window_id'].values
        df_pca_tmp['filename'] = df_features_tmp['filename'].values
        df_pca_tmp['file_id'] = df_features_tmp['file_id'].values
    
        # add  cluster IDs
        df_pca_tmp['cluster_id'] = kmeans.labels_
        df_pca.append(df_pca_tmp)

        # cluster centers
        df_cluster_centers_tmp = pd.DataFrame(
            kmeans.cluster_centers_, columns=pca_columns
        )

        df_cluster_centers_tmp['frequency_window'] = [df_features_tmp['frequency_window'].iloc[0]] * len(df_cluster_centers_tmp)
        df_cluster_centers_tmp['frequency_window_id'] = df_features_tmp['frequency_window_id'].iloc[0]
    
        df_cluster_centers.append(df_cluster_centers_tmp)
        

    # Concatenate DataFrame in lists to single DataFrames
    df_pca = pd.concat(df_pca, ignore_index=True)
    df_cluster_centers = pd.concat(df_cluster_centers, ignore_index=True)
    
    return df_pca, df_cluster_centers


# %%
def sample_files_by_hour(source_dir, destination_dir, n_files):
    """
    Samples exactly n_files files for each hour from a directory of sound recordings
    based on filenames in the format 'SM15XPRIZE_YYYYMMDD_HHMMSS.wav', and copies
    them to a separate destination directory. Cleans the destination directory if it exists.

    Parameters:
    - source_dir (str): Path to the directory containing the sound recordings.
    - destination_dir (str): Path to the directory where sampled files will be copied.

    Returns:
    - None
    """

    # Check if destination directory exists
    if os.path.exists(destination_dir):
        # Clean destination directory
        shutil.rmtree(destination_dir)
        print(f"Cleaning existing directory: {destination_dir}")

    # Ensure the destination directory exists
    os.makedirs(destination_dir)

    # Dictionary to store files by hour
    files_by_hour = defaultdict(list)

    # Collect files by hour
    for filename in os.listdir(source_dir):
        if filename.endswith('.wav'):  # Assuming files have .wav extension
            # Extract hour from filename
            try:
                hour = filename.split('_')[2][-10:-8]  # Extract the hour part
                files_by_hour[hour].append(filename)
            except IndexError:
                print(f"Filename format unexpected: {filename}")

    # List to store sampled files
    sampled_files = []

    # Sample exactly 2 files per hour
    for hour in sorted(files_by_hour.keys()):  # Ensure hours are processed in order
        files = files_by_hour[hour]
        if len(files) > 0:
            sampled_files.extend(random.sample(files, min(n_files, len(files))))  # Sample up to 3 files

    # Copy sampled files to destination directory
    for file in sampled_files:
        source_path = os.path.join(source_dir, file)
        destination_path = os.path.join(destination_dir, file)
        shutil.copy2(source_path, destination_path)

    print(f"{len(sampled_files)} files have been sampled and copied successfully.")

def plot_hour_distribution(directory):
    """
    Plots a histogram of the distribution of hours from filenames in the specified directory.

    Parameters:
    - directory (str): Path to the directory containing the sound recordings.

    Returns:
    - None
    """
    # List to store hours
    hours = []

    # Collect hours from filenames
    for filename in os.listdir(directory):
        if filename.endswith('.wav') and filename.startswith('SM15XPRIZE_'):
            try:
                hour = filename.split('_')[2][-10:-8]
                hours.append(int(hour))  # Convert hour to integer
            except IndexError:
                print(f"Filename format unexpected: {filename}")

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(hours, bins=range(0, 24), edgecolor='black', alpha=0.7)
    plt.xticks(range(0, 24))
    plt.xlabel('Hour of the Day')
    plt.ylabel('Frequency')
    plt.title('Distribution of Hours in Directory')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.axhline(y=4)
    plt.show()

def extract_datetime_from_filename(filename):
    try:
        date_str = filename.split('_')[1]  # Extract 'YYYYMMDD' part
        time_str = filename.split('_')[2].split('.')[0]  # Extract 'HHMMSS' part and remove '.wav'
        dt = datetime.strptime(date_str + time_str, '%Y%m%d%H%M%S')
        return dt
    except IndexError:
        print(f"Filename format unexpected: {filename}")
        return None


# %%
def saveFeatures_Comp(feature_dir, features):

    pure_features = features["features"]

    df = pd.DataFrame(pure_features, columns=[f'var_{i+1}' for i in range(len(pure_features[0]))])

    df["time_window"] = features["time_window"]
    df["frequency_window"] = features["frequency_window"]

    fname = os.path.join(feature_dir, "current_features.csv")

    df.to_csv(fname)

def saveFeatures_Clustering(feature_dir, name, df_features, df_pca, df_mean_pca_by_cluster):

    fname_features = os.path.join(feature_dir, "f_" + name + ".csv")
    fname_pca = os.path.join(feature_dir, "p_" + name + ".csv")
    fname_mean = os.path.join(feature_dir, "m_" + name + ".csv")

    df_features.to_csv(fname_features)
    df_pca.to_csv(fname_pca)
    df_mean_pca_by_cluster.to_csv(fname_mean)    


# %% [markdown]
# # config.

# %%
# time sampling parameters
w_dt = 1.0 # time window of each sample [sec]
w_dt_shift = 1.0 # time by which samples are shifted [sec]

# frequency sampling parameters
w_df = 1000 # ferquency indow of each sample [Hz]
w_df_shift = 1000 # ferquency by which windows are shifted [Hz]

# fft parameters
n_fft = 2048 # number of sampling points in sub-samples used for fft (sets time resolution of spectra)
n_fft_shift = 128 # number of sampling points by which fft sub-samples are shifted

# freq_min, freq_max =  0.0, 48000.0 # min/max frequency of spectra [Hz]
freq_min, freq_max =  1000.0, 10000.0 # min/max frequency of spectra [Hz]

n_filters = 16
kernel_size = 6
padding = 4

n_clusters_kmeans = 5
n_pca_components = 20

max_files = 2

audio_file_dir = '/python/bird_sounds'
# audio_file_dir = '/python/data/SoundMeters_Ingles_Primary-20240519T132658Z-009/SoundMeters_Ingles_Primary/'
plot_dir = '/python/plots/'
feature_dir = '/python/features/'

# Two feature directories, one for the clustering features and one for the comparison features
feature_dir_cl = "../data/features_cluster"
feature_dir_comp = "../data/features_comp"
# audio_dir_pth = "../data/audio_files"
# audio_pth = "../data/audio_files/20231106_143000.WAV"
audio_dir_pth = "../bird_sounds/"
audio_pth = "../bird_sounds/SM15XPRIZE_20240411_160702.wav"

audio_dir_pth = '/python/data/audio_files_24h'
audio_dir_pth_subsample = '/python/data/24h_audio_files_subsample'

# %%
# sample 3 files for each hoer of the day and copy them into separate directory
sample_files_by_hour(audio_dir_pth, audio_dir_pth_subsample, 3)

# plot number of files per hour
plot_hour_distribution(audio_dir_pth)

# %%
name = "my-cluster-1"
t0 = time.time()
# single_audio_features = extractFeaturesFromFile(audio_pth, feature_dir_comp, w_dt, w_dt_shift, w_df, w_df_shift, n_fft, n_fft_shift, freq_min, freq_max, n_clusters_kmeans, n_pca_components)
df_features, correlation_matrix = extractFeaturesFromFolder(audio_dir_pth_subsample, feature_dir_cl, max_files, w_dt, w_dt_shift, w_df, w_df_shift, n_fft, n_fft_shift, freq_min, freq_max, n_filters, kernel_size, padding)
t1 = time.time() - t0

# %%
df_pca, df_pca_cluster_centers = get_pca_clustering(n_clusters_kmeans, n_pca_components, df_features)

# %% [markdown]
# # get number of clusters per sound file

# %%
df_pca['datetime'] = df_pca['filename'].apply(extract_datetime_from_filename)

clusters_per_band = (df_pca.groupby(['datetime', 'frequency_window_id', 'filename'])
                     ['cluster_id'].nunique()
                     .reset_index(name='n_cluster'))

freq_window_mapping = df_pca.groupby('frequency_window_id')['frequency_window'].apply(lambda x: x.iloc[0]).reset_index()

# Merge the frequency window information back into clusters_per_band
clusters_per_band = clusters_per_band.merge(freq_window_mapping, on='frequency_window_id', how='left')


clusters_per_band['time_of_day'] = clusters_per_band['datetime'].dt.time

# %%
# Plotting
plt.figure(figsize=(8, 6))

# Iterate over unique frequency_window_id
for freq_id in clusters_per_band['frequency_window_id'].unique():
    freq_band_data = clusters_per_band[clusters_per_band['frequency_window_id'] == freq_id]
    
    # Convert time_of_day to hours for plotting (alternative approach)
    time_hours = [t.hour + t.minute/60 + t.second/3600 for t in freq_band_data['time_of_day']]
    
    plt.scatter(time_hours, freq_band_data['n_cluster'], marker='o', label=f'Frequency Band {freq_id}')

plt.title('Number of Unique Clusters per Frequency Band vs Time of Day')
plt.xlabel('Time of Day (hours)')
plt.ylabel('Number of Unique Clusters')

# Move legend outside the plot
plt.legend(title='Frequency Band ID', loc='center left', bbox_to_anchor=(1, 0.5))

plt.grid(True)
plt.tight_layout()
plt.show()

# %%

fontsize=18

# Get the unique frequency window IDs
unique_freq_ids = clusters_per_band['frequency_window_id'].unique()

# Create subplots
fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharex=True, sharey=True)

# Flatten the 3x3 array of axes for easier indexing
axes_flat = axes.flatten()

# Iterate over unique frequency_window_id and create a subplot for each
for i, freq_id in enumerate(unique_freq_ids):
    freq_band_data = clusters_per_band[clusters_per_band['frequency_window_id'] == freq_id]
    
    # Convert time_of_day to hours for plotting
    time_hours = [t.hour + t.minute/60 + t.second/3600 for t in freq_band_data['time_of_day']]
    
    ax = axes_flat[i]
    ax.scatter(time_hours, freq_band_data['n_cluster'], marker='o', c='k')
    
    # Extract frequency window bounds
    frequency_window = freq_band_data["frequency_window"].iloc[0]
    lower_bound, upper_bound = frequency_window[0], frequency_window[1]
    
    ax.set_title(f'{lower_bound} < f < {upper_bound} Hz',fontsize=fontsize)
    ax.grid(True)

    # Set tick label fontsize
    plt.setp(ax.get_xticklabels(), fontsize=fontsize)
    plt.setp(ax.get_yticklabels(), fontsize=fontsize)

# Hide unused subplots if there are any
for j in range(i + 1, len(axes_flat)):
    fig.delaxes(axes_flat[j])

# Set common labels
fig.text(0.5, 0.04, 'Time of Day (hours)', ha='center', va='center', fontsize=fontsize)
fig.text(0.04, 0.5, 'Number of Unique Clusters', ha='center', va='center', rotation='vertical',fontsize=fontsize)

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0.05, 0.05, 1, 1])

# Show the plot
plt.show()

# %% [markdown]
# # plots for poster

# %% [markdown]
# ### spectrum vs clusters for single file

# %%
audio_dir_pth_subsample

# %%
filename

# %%
# select audio file
file_id = 0

filename = df_pca[df_pca.file_id==file_id].filename.unique()[0]
df_pca_file = df_pca[df_pca.filename==filename]

# %%
# get spectrum
audio_file_path = audio_dir_pth + '/'+ filename

audio_file_path, features, time_windows, frequency_windows = process_audio_file(
    audio_file_path,
    freq_min, freq_max,
    w_df, w_df_shift,
    w_dt, w_dt_shift,
    n_fft, n_fft_shift,
    n_filters, kernel_size, padding
)


# Get spectrogram
amp, sr = librosa.load(audio_file_path, sr=None)  # Load amplitude and sampling rate from file
spec = librosa.stft(amp, n_fft=n_fft, hop_length=n_fft_shift)
spec_db = librosa.amplitude_to_db(np.abs(spec))

# Set time windows
n_sec_in_audio_file = len(amp) / sr
time_windows = get_windows(0, n_sec_in_audio_file, w_dt, w_dt_shift)

print(frequency_windows)

# %%
# plot

# fig, ax = plt.subplots(len(frequency_windows)+1,1, sharex=True, figsize=(10,10), gridspec_kw={'height_ratios': [1,1,1,3]})
fig, ax = plt.subplots(len(frequency_windows)+1,1, sharex=True, figsize=(10,10))


# ======== plot cluster ==========
# Loop over each frequency ID and plot the cluster data
for freq_id, frequency_window in enumerate(frequency_windows):

    iy = freq_id
    
    df_pca_freq = df_pca_file[df_pca_file.frequency_window_id == freq_id]

    time_windows = df_pca_freq["time_window"].to_numpy()
    absolute_times = [np.mean(w) for w in time_windows]

    ax[iy].scatter(absolute_times, df_pca_freq.cluster_id)
    ax[iy].set_ylabel(f'(freq. band{freq_id})')

    # Add vertical lines for each time window
    for lo, hi in time_windows:
        ax[iy].axvline(x=lo, ls='-', c='r', lw=0.5)
        ax[iy].axvline(x=hi, ls='-', c='r', lw=0.5)


ax[0].set_title(f'cluster IDs ({filename}), kernel size: {kernel_size}, padding: {padding}')
ax[-1].set_title('Sound Spectrogram')

# ========= plot spectrum ==========

librosa.display.specshow(spec_db, sr=sr, x_axis='time', y_axis='linear', cmap='rainbow', ax=ax[-1], n_fft=n_fft, hop_length=n_fft_shift)

for lo, hi in time_windows:
    ax[-1].axvline(x=lo, ls='-', c='r', lw=0.5)
    ax[-1].axvline(x=hi, ls='-', c='r', lw=0.5)
    
for i, (lo, hi) in enumerate(frequency_windows):
    plt.axhline(y=lo,ls='-',c='r', lw=0.5)
    plt.axhline(y=hi,ls='-',c='r', lw=0.5)

# ax[0].set_title(filename)
ax[-1].set_ylim(freq_min, freq_max)

plt.tight_layout()

# %%
# plot
from matplotlib.ticker import MaxNLocator

fig, ax = plt.subplots(2,1, sharex=True, figsize=(10,8), gridspec_kw={'height_ratios': [3, 1]})


fontsize = 16

freq_id = 2

# ========= plot spectrum ==========
iy = 0

# ax[iy].set_title(f'Sound Spectrogram ({filename})')
ax[iy].set_title(f'Sound Spectrogram')

librosa.display.specshow(spec_db, sr=sr, x_axis='time', y_axis='linear', cmap='rainbow', ax=ax[0], n_fft=n_fft, hop_length=n_fft_shift)

ax[0].axhline(y=frequency_windows[freq_id][0],c='r', lw=2)
ax[0].axhline(y=frequency_windows[freq_id][1],c='r', lw=2)


ax[0].set_ylim(freq_min, freq_max)

# ======== plot cluster ==========
df_pca_freq = df_pca_file[df_pca_file.frequency_window_id == freq_id]

time_windows = df_pca_freq["time_window"].to_numpy()
absolute_times = [np.mean(w) for w in time_windows]

ax[1].scatter(absolute_times, df_pca_freq.cluster_id, c='k')


ax[1].set_ylim(0, df_pca_freq.cluster_id.max()+0.5)

# ======== time windows ======
for iy in range(2):
    for lo, hi in time_windows:
        ax[iy].axvline(x=lo, ls='-', c='r', lw=1)
        ax[iy].axvline(x=hi, ls='-', c='r', lw=1)

# set tick size
for iy in range(2):
    ax[iy].tick_params(axis='both', which='major', labelsize=14, length=10, width=2)
    ax[iy].tick_params(axis='both', which='minor', labelsize=14, length=10, width=2)

ax[0].yaxis.set_major_locator(MaxNLocator(integer=True))
ax[1].yaxis.set_major_locator(MaxNLocator(integer=True))

ax[0].set_xlabel('')
ax[0].set_ylabel('frequency [Hz]', fontsize=fontsize)

ax[1].set_xlabel('time [sec]', fontsize=fontsize)
ax[1].set_ylabel('cluster ID', fontsize=fontsize)


plt.tight_layout()

# %%
df_pca_freq = df_pca_file[df_pca_file.frequency_window_id == freq_id]

# %%
frequency_windows

# %%
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.patches as patches

# Create subplots with shared x-axis and specified height ratios
fig, ax = plt.subplots(3, 1, sharex=True, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1, 1]})

librosa.display.specshow(spec_db, sr=sr, x_axis='time', y_axis='linear', cmap='Greys', ax=ax[0], n_fft=n_fft, hop_length=n_fft_shift)

# Parameters
fontsize = 16
freq_ids = [2, 0]
freq_colors = ['r','b']

# Plot the spectrum
# ax[0].set_title('Sound Spectrogram')
ax[0].set_ylim(freq_min, freq_max)

# Highlight frequency windows with shaded areas
for i, freq_id in enumerate(freq_ids):
    tmin = np.min(time_windows)[0]
    tmax = np.max(time_windows)[1]
    
    ax[0].fill_betweenx(y=[frequency_windows[freq_id][0], frequency_windows[freq_id][1]], 
                        x1=tmin, x2=tmax, color=freq_colors[i], alpha=0.2)

# Plot clusters
for i, freq_id in enumerate(freq_ids):
    df_pca_freq = df_pca_file[df_pca_file.frequency_window_id == freq_id]
    time_windows = df_pca_freq["time_window"].to_numpy()
    absolute_times = [np.mean(w) for w in time_windows]
    
    ax[i + 1].scatter(absolute_times, df_pca_freq.cluster_id, color=freq_colors[i])
    ax[i + 1].set_ylim(0, df_pca_freq.cluster_id.max() + 0.5)
    ax[i + 1].set_ylabel('cluster ID', fontsize=fontsize)

    # Add label with frequency windows to bottom panels
    freq_range = f'{frequency_windows[freq_ids[i]][0]} < f < {frequency_windows[freq_ids[i]][1]} Hz'
    ax[i + 1].text(0.05, 0.05, freq_range, transform=ax[i + 1].transAxes, fontsize=fontsize,
                   bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'), ha='left', va='bottom')

    # Manually set y-axis ticks on the bottom panels to intervals of 2
    y_ticks = np.arange(0, df_pca_freq.cluster_id.max() + 2, 2)
    ax[i + 1].set_yticks(y_ticks)
    
# Add white box to the bottom right of the panels
rect = patches.Rectangle((0.8, 0.05), 0.15, 0.1, linewidth=1, edgecolor='none', facecolor='white', alpha=0.5)
ax[1].add_patch(rect)


# Add vertical lines for time windows across all subplots
for subplot in ax:
    for lo, hi in time_windows:
        subplot.axvline(x=lo, linestyle='-', color='grey', linewidth=0.5)
        subplot.axvline(x=hi, linestyle='-', color='grey', linewidth=0.5)

# Customize tick parameters and axis labels
for subplot in ax:
    subplot.tick_params(axis='both', which='major', labelsize=14, length=10, width=2)
    subplot.tick_params(axis='both', which='minor', labelsize=14, length=10, width=2)

# Set axis labels
ax[0].set_xlabel('')
ax[0].set_ylabel('frequency [Hz]', fontsize=fontsize)
ax[-1].set_xlabel('time [sec]', fontsize=fontsize)

# Adjust layout
plt.tight_layout()


# %%
df_pca_freq = df_pca[df_pca.frequency_window_id == 4]

# %%
df_pca_freq.frequency_window.iloc[0]

# %%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

cmap = 'rainbow'

# Define the dimensions for the scatter plot matrix
dimensions = ['PCA_1', 'PCA_2', 'PCA_3']

# Sort cluster IDs and define a consistent palette
unique_clusters = sorted(df_pca_freq['cluster_id'].unique())
palette = sns.color_palette(cmap, len(unique_clusters))
palette_dict = {cluster: color for cluster, color in zip(unique_clusters, palette)}

# Create pair plot without the automatic legend
pair_plot = sns.pairplot(df_pca_freq, vars=dimensions, hue='cluster_id', palette=palette_dict, diag_kind=None)
pair_plot._legend.remove()  # Remove the default legend

# Remove diagonals and top-right panels
for i in range(len(dimensions)):
    for j in range(len(dimensions)):
        if i <= j:
            pair_plot.axes[i, j].set_visible(False)

# Add labels for x-axis and y-axis
for i, dim1 in enumerate(dimensions):
    for j, dim2 in enumerate(dimensions):
        if i > j:
            pair_plot.axes[i, j].set_xlabel(dim2)
            pair_plot.axes[i, j].set_ylabel(dim1)

# Create dummy handles for the legend
handles = [plt.Line2D([], [], marker='o', color=palette_dict[cluster], linestyle='', markersize=8) for cluster in unique_clusters]
labels = [f'Cluster {cluster}' for cluster in unique_clusters]

# Add legend to the top-right panel
pair_plot.fig.legend(handles=handles, labels=labels, title='', loc='upper right', bbox_to_anchor=(0.6, 0.65))
pair_plot.axes[0, 2].set_axis_off()

# Adjust layout
plt.subplots_adjust(wspace=0.3, hspace=0.3)

# Set the title
plt.title('Pairwise Plot of PCA Components with Cluster Analysis')

freq_range = df_pca_freq.frequency_window.iloc[0]

freq_limits = f'{freq_range[0]} < f < {freq_range[1]} Hz'

pair_plot.axes[0, 0].set_title(freq_limits)

plt.savefig('pairwise_plot.png', dpi=300)
# print(freq_limits)

plt.show()


# %% [markdown]
# ### make sure there is no invalid window processing going on 

# %%
none_count, none_files = count_none_frequency_windows(features_all_files)
print(f"Number of dictionaries with None frequency_windows: {none_count}")
print("Files with None frequency_windows:", none_files)

# %%
# Plot the histogram of all feature values
plt.hist(df_features.iloc[:, 0], alpha=0.5)
plt.xlabel('Feature Value')
plt.ylabel('Frequency')
plt.title('Distribution of Feature Values Across All Files')
plt.show()

# %% [markdown]
# ## plot correlation of features with the highest values

# %%
# Identify the features with the highest values
# Here we assume "highest values" means the highest mean value for simplicity
# Exclude non-numeric columns
numeric_df_features = df_features.select_dtypes(include=[np.number])

# Calculate the mean values of each column (feature)
mean_values = numeric_df_features.mean()

# Identify the top 100 features with the highest mean values
top_features = mean_values.nlargest(100).index

# Keep only these columns in the DataFrame
df_features_reduced = numeric_df_features[top_features]

# Calculate the correlation matrix
correlation_matrix = df_features_reduced.corr()

# Plot the correlation matrix using a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix of Top 100 Features with Highest Mean Values')
plt.show()

# %%
df_pca

# %%
# Create a pair plot using Plotly
fig = px.scatter_matrix(
    df_pca,
    dimensions=[f'PCA_{i+1}' for i in range(n_pca_components)],
    color='cluster_id',
    hover_data={
        'filename': True,
        'time_window': True,
    },
    title='Pairwise Plot of PCA Components'
)

# Update layout for better visibility
fig.update_layout(
    width=1200,
    height=1200,
    dragmode='select',
    hovermode='closest', 
    coloraxis_showscale=False  # Disable color scale bar
)

# Show the plot
fig.show()

# %% [markdown]
# ### trying to add labels to the features for reproducing sounds

# %%
import librosa
import numpy as np
from IPython.display import Audio
import os

time_window = (50.5, 51.0) # Convert the time window to sample indices
# Load the audio file
#filename = 'SM4XPRIZE_20240410_114402.wav' # use this to specify a file that you want to check from the clustering (hover over the plot above)
filename = os.listdir(audio_file_dir)[0] # use this if you just want the first file in the directory

# %%
audio_file_path = os.path.join(audio_file_dir, filename)
y, sr = librosa.load(audio_file_path, sr=None)

# Convert the time window to sample indices
start_sample = int(time_window[0] * sr)
end_sample = int(time_window[1] * sr)

# Extract the corresponding segment from the audio file
y_segment = y[start_sample:end_sample]

# Debugging: Check the extracted segment
print("Extracted audio segment:")
print(y_segment)

# %%
# Check if you can play the original segment
print("Playing segment:")
Audio(y_segment, rate=sr)



# %% [markdown]
# ## write features values into output file
#
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# filename = feature_dir + f'sound_features_pca_{timestamp}.csv'
# df_pca.to_csv(filename)
#
#

# %% [markdown]
# ## get cluster centers

# %%
# Identify PCA columns
pca_columns = [col for col in df_pca.columns if col.startswith('PCA')]

# Compute the mean of all PCA columns for each cluster
mean_pca_values_by_cluster = df_pca.groupby('cluster_id')[pca_columns].mean()

# %%
cols = ['PCA_1','PCA_2']

plt.scatter(df_pca[cols[0]], df_pca[cols[1]], c=df_pca['cluster_id'],cmap='jet')
plt.scatter(mean_pca_values_by_cluster[cols[0]], mean_pca_values_by_cluster[cols[1]], c='k',marker='x',s=100)

plt.xlabel(cols[0])
plt.ylabel(cols[1])


# %%
def generate_filename(method='kmeans', n_clusters=None):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if method.lower() == 'kmeans' and n_clusters is not None:
        filename = f'cluster_centers_{method}_Nclust{n_clusters}_{timestamp}.csv'
    else:
        filename = f'mean_pca_values_{method}_{timestamp}.csv'
    return filename



# %%
# Generate a meaningful filename
filename = feature_dir + generate_filename('kmeans',n_clusters_kmeans)

# Write the mean PCA values by cluster to a CSV file
mean_pca_values_by_cluster.to_csv(filename)


# plot cluster vs time (in hours), only works if filename is written as name_yyyymmdd_hhmmss.wav
# plot_cluster_vs_absolute_time(df_pca)
