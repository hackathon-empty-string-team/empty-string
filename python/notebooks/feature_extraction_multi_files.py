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
import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import tensorflow as tf


# %%
# array containing lower and upper limits of time windows
def get_windows(val_min,val_max,window_width,window_shift):
    windows = []
    n_bin = int((val_max - val_min) / window_shift)
    for i in range(n_bin):
        lo = i*window_shift
        hi = lo + window_width
        windows.append([lo, hi])
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

    Parameters:
    S_db_mod (numpy.ndarray): Modified spectrogram in dB.

    Returns:
    numpy.ndarray: Output vector after Conv2D and global average pooling.
    """
    # Reshape to add batch dimension (1, height, width, channels)
    input_tensor = tf.reshape(S_db_mod, (1, S_db_mod.shape[0], S_db_mod.shape[1], 1))
    
    # Create a Conv2D layer with 128 filters, 3x3 kernel size, and 'same' padding
    conv_layer = tf.keras.layers.Conv2D(
        filters=256,
        kernel_size=(10, 10),
        padding='same',
        activation='relu',
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42)
    )
    
    # Apply the Conv2D layer to the input tensor
    output_tensor = conv_layer(input_tensor)
    
    # Apply global average pooling to get the output as a vector with 128 features
    output_vector = tf.keras.layers.GlobalAveragePooling2D()(output_tensor)
    
    # Convert the output vector to a numpy array
    output_vector_np = output_vector.numpy()
    
    return output_vector_np

def get_features(spec_db, times, time_windows, frequencies, frequency_windows):
    # Loop through each combination of frequency and time windows
    # for each time window append all features
    # found in the different frequency windoes into one list
    
    features = []
    
    for i, (tlo, thi) in enumerate(time_windows):

        if i % 20==0: print('time window ', i)
    
        features_in_time_window = []
        
        for j, (flo, fhi) in enumerate(frequency_windows):
            
            t_mask = (times >= tlo) & (times <= thi)
            f_mask = (frequencies >= flo) & (frequencies <= fhi)
        
            # Extract the subsample based on the masks
            subsample = spec_db[f_mask][:, t_mask]
        
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
    - features: Extracted features.
    """
    # Read the audio file
    amp, sr = librosa.load(audio_file_path, sr=None)  # Load amplitude and sampling rate from file

    # Get spectrogram
    spec = librosa.stft(amp, n_fft=n_fft, hop_length=n_fft_shift)
    spec_db = librosa.amplitude_to_db(np.abs(spec))


    # Set time windows
    n_sec_in_audio_file = len(amp) / sr
    time_windows = get_windows(0, n_sec_in_audio_file, w_dt, w_dt_shift)

  
    # Set frequency windows
    frequency_windows = get_windows(freq_min, freq_max, w_df, w_df_shift)
    
    # plot_spectrogram_with_windows(spec_db, sr, time_windows, frequency_windows, n_fft, n_fft_shift)
    
    times = librosa.frames_to_time(np.arange(spec_db.shape[1]), sr=sr, n_fft=n_fft, hop_length=n_fft_shift)
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # Feature extraction
    features = get_features(spec_db, times, time_windows, frequencies, frequency_windows)
    
    return features

# %% [markdown]
# # config

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

audio_file_dir = '/python/data/'
plot_dir = '/python/plots/'

# %% [markdown]
# # get features from all wav files in directory

# %%
import time

# %%
features_all_files = []

# Loop over filenames in the directory
for filename in os.listdir(audio_file_dir):
    if filename.endswith(".wav"):  # Example: process only .wav files
        file_path = os.path.join(audio_file_dir, filename)
        print('processing ',filename)

        start_time = time.time()
        
        features = process_audio_file(file_path,
                       freq_min, freq_max,
                       w_df, w_df_shift,
                       w_dt, w_dt_shift,
                       n_fft, n_fft_shift
                    )

        features_all_files = features_all_files + features

        execution_time = time.time() - start_time
        
        print(f"Execution time: {execution_time:.2f} seconds \n")

# %%
# convert to numpy array
features_all_files = np.array(features_all_files)

# %%
plt.hist(features_all_files[10], bins=np.linspace(0,11,100),alpha=0.5)
plt.xlabel('N')
plt.xlabel('feature value')
plt.title('distribution of feature values')
plt.show()

# %% [markdown]
# # plot correlation of features with the highest values

# %%
import pandas as pd
import seaborn as sns

# %%
df_features = pd.DataFrame(features_all_files, columns=[f'Var_{i+1}' for i in range(len(features_all_files[0]))])

# %%
# Calculate the average (mean) of each column
column_means = df_features.mean()

# Identify the N columns with the highest average values
columns_to_keep = column_means.nlargest(100).index

# Keep only these columns in the DataFrame
df_features_reduced = df_features[columns_to_keep]

correlation_matrix = df_features_reduced.corr()

# %%
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# %% [markdown]
# # kmeans clustering

# %%
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# %%
# Perform KMeans clustering
kmeans = KMeans(n_clusters=15, random_state=0).fit(features_all_files_normalized)
labels = kmeans.labels_

# %% [markdown]
# # plot PCA components

# %%
import seaborn as sns

n_pca_components = 6

# Assuming features_all_files and labels are already defined
pca = PCA(n_components=n_pca_components)
features_pca = pca.fit_transform(features_all_files)

# Create a DataFrame from the PCA features
df = pd.DataFrame(features_pca, columns=[f'PCA_{i+1}' for i in range(n_pca_components)])

# Add labels to the DataFrame
df['Cluster'] = labels

# Plot pairwise relationships in the DataFrame
sns.pairplot(df, hue='Cluster', palette='Set1', diag_kind='kde', plot_kws={'alpha': 0.6, 's': 5}, height=2)

plt.savefig(plot_dir+'pairplot.png')
plt.show()


# %%
