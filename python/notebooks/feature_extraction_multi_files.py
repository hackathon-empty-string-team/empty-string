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
from sklearn.cluster import KMeans
import plotly.express as px
from datetime import datetime


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

def apply_conv2d(S_db_mod):
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
        out_channels=256,
        kernel_size=10,
        padding=5,
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


def get_features(spec_db, times, time_windows, frequencies, frequency_windows):
    features = []
    
    for i, (tlo, thi) in enumerate(time_windows):
    
        features_in_time_window = []
        
        for j, (flo, fhi) in enumerate(frequency_windows):

            assert tlo != thi, "Tlo is equal thi"
            
            t_mask = (times >= tlo) & (times <= thi)
            f_mask = (frequencies >= flo) & (frequencies <= fhi)

            #print(flo, fhi, frequencies)
        
            # Extract the subsample based on the masks
            subsample = spec_db[f_mask][:, t_mask]

            #normalize the subsample with the absolute maximum value
            subsample = subsample / np.max(np.abs(subsample))

            #print(subsample.shape)
        
            output_vector_np = apply_conv2d(subsample)
        
            features_in_time_window.append(output_vector_np)
    
        features.append(np.array(features_in_time_window).flatten())

    return features


def process_audio_file(audio_file_path, freq_min, freq_max, w_df, w_df_shift, w_dt, w_dt_shift, n_fft, n_fft_shift):
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
    features = get_features(spec_db, times, time_windows, frequencies, frequency_windows)
    
    
    return audio_file_path, features, time_windows, frequency_windows

def process_multiple_audiofiles(audio_file_dir, max_files, freq_min, freq_max, w_df, w_df_shift, w_dt, w_dt_shift, n_fft, n_fft_shift):
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
                n_fft, n_fft_shift
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
# ## config

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

audio_file_dir = '/python/data/audio_files'
# audio_file_dir = '/python/data/SoundMeters_Ingles_Primary-20240519T132658Z-009/SoundMeters_Ingles_Primary/'
plot_dir = '/python/plots/'

feature_dir = '/python/features/'

max_files = 4

n_clusters_kmeans = 10
n_pca_components = 8


# %% [markdown]
# ## feature extraction & clustering functions

# %%
def extractFeaturesFromFile(audio_pth, feature_dir, w_dt, w_dt_shift, w_df, w_df_shift, n_fft, n_fft_shift, freq_min, freq_max, n_clusters_kmeans, n_pca_components):
    plot_dir = '/python/plots/'
    feature_dir = '/python/features/'

    audio_file_path, features, time_windows, frequency_windows = process_audio_file(
                audio_pth,
                freq_min, freq_max,
                w_df, w_df_shift,
                w_dt, w_dt_shift,
                n_fft, n_fft_shift
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
def extractFeaturesFromFolder(audio_dir_pth, feature_dir, w_dt, w_dt_shift, w_df, w_df_shift, n_fft, n_fft_shift, freq_min, freq_max, n_clusters_kmeans, n_pca_components):

    max_files = 2
    features_all_files = process_multiple_audiofiles(audio_file_dir, max_files, freq_min, freq_max, w_df, w_df_shift, w_dt, w_dt_shift, n_fft, n_fft_shift)

    # Initialize empty lists to collect all feature values and corresponding metadata
    all_features = []
    all_time_windows = []
    all_filenames = []
    
    # Loop through the list of dictionaries and collect feature values and metadata
    for file_data in features_all_files:
        filename = file_data["filename"]
        features = file_data["features"]
        time_windows = file_data["time_windows"]
    
        for i, feature_array in enumerate(features):
            all_features.append(feature_array)  # Collect feature arrays
            all_time_windows.append(time_windows[i] if i < len(time_windows) else None)
            all_filenames.append(filename)
    
    # Convert the list of feature arrays into a single numpy array
    all_features = np.vstack(all_features)
    
    # Convert the numpy array to a pandas DataFrame
    df_features = pd.DataFrame(all_features, columns=[f'var_{i+1}' for i in range(len(all_features[0]))])
    
    # Add the time windows and filenames to the DataFrame
    df_features['time_window'] = all_time_windows
    df_features['filename'] = all_filenames

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

    return df_features, all_features, correlation_matrix
        


# %%
def getClustering(n_clusters_kmeans, n_pca_components, raw_features, df_features):

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters_kmeans, random_state=0).fit(raw_features)
    labels = kmeans.labels_
    
    # Apply PCA on the features
    pca = PCA(n_components=n_pca_components)
    features_pca = pca.fit_transform(raw_features)
    
    # Create a DataFrame from the PCA features
    df_pca = pd.DataFrame(features_pca, columns=[f'PCA_{i+1}' for i in range(n_pca_components)])
    
    # Add the time windows, frequency windows, filenames, and cluster labels to the DataFrame
    df_pca['time_window'] = df_features['time_window']
    df_pca['filename'] = df_features['filename']
    df_pca['Cluster'] = labels

    # Identify PCA columns
    pca_columns = [col for col in df_pca.columns if col.startswith('PCA')]
    
    # Compute the mean of all PCA columns for each cluster
    mean_pca_values_by_cluster = df_pca.groupby('Cluster')[pca_columns].mean()

    return df_pca, mean_pca_values_by_cluster


# %%
def saveFeatures_Comp(feature_dir, features):

    pure_features = features["features"]

    df = pd.DataFrame(pure_features, columns=[f'var_{i+1}' for i in range(len(pure_features[0]))])

    df["time_window"] = features["time_windows"]

    fname = os.path.join(feature_dir, "current_features.csv")

    df.to_csv(fname)

def saveFeatures_Clustering(feature_dir, name, df_features, df_pca, df_mean_pca_by_cluster):

    fname_features = os.path.join(feature_dir, "f_" + name + ".csv")
    fname_pca = os.path.join(feature_dir, "p_" + name + ".csv")
    fname_mean = os.path.join(feature_dir, "m_" + name + ".csv")

    df_features.to_csv(fname_features)
    df_pca.to_csv(fname_pca)
    df_mean_pca_by_cluster.to_csv(fname_mean)    


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
df_pca, mean_pca_values_by_cluster = getClustering(n_clusters_kmeans, n_pca_components, raw_features, df_features)

# %%
saveFeatures_Comp(feature_dir_comp, single_audio_features)

# %%
saveFeatures_Clustering(feature_dir_cl, name, df_features, df_pca, mean_pca_values_by_cluster)

# %% [markdown]
# # Notebook code

# %%
features_all_files = process_multiple_audiofiles(audio_file_dir, max_files, freq_min, freq_max, w_df, w_df_shift, w_dt, w_dt_shift, n_fft, n_fft_shift)

# %% [markdown]
# ### make sure there is no invalid window processing going on 

# %%
none_count, none_files = count_none_frequency_windows(features_all_files)
print(f"Number of dictionaries with None frequency_windows: {none_count}")
print("Files with None frequency_windows:", none_files)

# %% [markdown]
# ## look at distribution of all feature values 

# %%
# Initialize empty lists to collect all feature values and corresponding metadata
all_features = []
all_time_windows = []
all_filenames = []

# Loop through the list of dictionaries and collect feature values and metadata
for file_data in features_all_files:
    filename = file_data["filename"]
    features = file_data["features"]
    time_windows = file_data["time_windows"]

    for i, feature_array in enumerate(features):
        all_features.append(feature_array)  # Collect feature arrays
        all_time_windows.append(time_windows[i] if i < len(time_windows) else None)
        all_filenames.append(filename)

# Convert the list of feature arrays into a single numpy array
all_features = np.vstack(all_features)

# Convert the numpy array to a pandas DataFrame
df_features = pd.DataFrame(all_features, columns=[f'var_{i+1}' for i in range(len(all_features[0]))])

# Add the time windows and filenames to the DataFrame
df_features['time_window'] = all_time_windows
df_features['filename'] = all_filenames

# %%
df_features.info()

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

# %% [markdown]
# ### clustering and pca

# %%
# Perform KMeans clustering
kmeans = KMeans(n_clusters=n_clusters_kmeans, random_state=0).fit(all_features)
labels = kmeans.labels_

# Apply PCA on the features
pca = PCA(n_components=n_pca_components)
features_pca = pca.fit_transform(all_features)

# Create a DataFrame from the PCA features
df_pca = pd.DataFrame(features_pca, columns=[f'PCA_{i+1}' for i in range(n_pca_components)])

# Add the time windows, frequency windows, filenames, and cluster labels to the DataFrame
df_pca['time_window'] = all_time_windows
df_pca['filename'] = all_filenames
df_pca['Cluster'] = labels

# %%
# Create a pair plot using Plotly
fig = px.scatter_matrix(
    df_pca,
    dimensions=[f'PCA_{i+1}' for i in range(n_pca_components)],
    color='Cluster',
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
mean_pca_values_by_cluster = df_pca.groupby('Cluster')[pca_columns].mean()

# %%
cols = ['PCA_1','PCA_2']

plt.scatter(df_pca[cols[0]], df_pca[cols[1]], c=df_pca['Cluster'],cmap='jet')
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
#plot_cluster_vs_absolute_time(df_pca)
