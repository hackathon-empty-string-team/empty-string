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
# # Labeled Training Approach

# %% [markdown]
# ## Comfiguring the Dataset and the DataLoader

# %%
import os
import pandas as pd
from torchvision.io import read_image

# %%
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
import pickle
import tensorflow as tf
import random
import torch
import time
from torch.utils.data import Sampler, Dataset, DataLoader
import math
from torch.nn.utils.rnn import pad_sequence

"""
2024-05-25 19:51:34.086854: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2024-05-25 19:51:38.067901: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
"""


# %% [markdown]
# Here I took Jan's Transformation functions. HOWEVER, most of these transformations are commented out, due to performance issues, namely:
#
# - Gaussians to Spectrograms
# - Convolutions incl. PoolingLayer

# %%
def gaussian_2d(x, y, mean_x, mean_y, std_dev_x, std_dev_y):
    """
    Generates a 2D Gaussian mask.

    Parameters:
    x (numpy.ndarray): X coordinates.
    y (numpy.ndarray): Y coordinates.
    mean_x (float): Mean value for the Gaussian distribution along the x-axis.
    mean_y (float): Mean value for the Gaussian distribution along the y-axis.
    std_dev_x (float): Standard deviation for the Gaussian distribution along the x-axis.
    std_dev_y (float): Standard deviation for the Gaussian distribution along the y-axis.

    Returns:
    numpy.ndarray: 2D Gaussian mask.
    """
    gauss_x = np.exp(-0.5 * ((x - mean_x) / std_dev_x) ** 2)
    gauss_y = np.exp(-0.5 * ((y - mean_y) / std_dev_y) ** 2)
    return gauss_x * gauss_y

def apply_gaussian_to_spectrogram(S_db, x_means, y_means, std_dev_x, std_dev_y):
    """
    Applies Gaussian masks to a spectrogram and returns the results.

    Essentially what we discussed the other day: In order to not "hard" cut the signal in x (time) and y (Hz) direction, we just apply Gauss in both directions.
    For now i chose to do 5 in each direction. Therefore, every timestamp we analyze, gets 25 "blobs"

    Parameters:
    S_db (numpy.ndarray): Spectrogram in dB.
    x_means (numpy.ndarray): Array of mean values for the Gaussian distribution along the x-axis.
    y_means (numpy.ndarray): Array of mean values for the Gaussian distribution along the y-axis.
    std_dev_x (float): Standard deviation for the Gaussian distribution along the x-axis.
    std_dev_y (float): Standard deviation for the Gaussian distribution along the y-axis.

    Returns:
    dict: Dictionary containing the Gaussian-masked spectrograms and their parameters.
    """
    n_rows, n_cols = S_db.shape
    x = np.linspace(0, n_cols, n_cols)
    y = np.linspace(0, n_rows, n_rows)
    x_grid, y_grid = np.meshgrid(x, y)
    
    results = {}
    
    for x_idx, x_mean in enumerate(x_means):
        for y_idx, y_mean in enumerate(y_means):
            gaussian_mask = gaussian_2d(x_grid, y_grid, x_mean, y_mean, std_dev_x, std_dev_y)
            S_db_gaussian = S_db * gaussian_mask
            key = f'x_{x_idx}_y_{y_idx}'
            results[key] = {
                'S_db_gaussian': S_db_gaussian,
                'x_mean': x_mean,
                'y_mean': y_mean,
                'std_dev_x': std_dev_x,
                'std_dev_y': std_dev_y
            }
    
    return results

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
    conv_layer = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')
    
    # Apply the Conv2D layer to the input tensor
    #output_tensor = conv_layer(input_tensor)
    output_tensor = input_tensor
    
    # Apply global average pooling to get the output as a vector with 128 features
    #output_vector = tf.keras.layers.GlobalAveragePooling2D()(output_tensor)
    output_vector = tf.reduce_mean(output_tensor, axis=[-2, -1], keepdims=True)
    # Convert the output vector to a numpy array
    output_vector_np = output_vector.numpy()
    
    return output_vector_np

def plot_random_spectrograms(results_dict):
    """
    Plots all modified spectrograms for a random time stamp. (Just for visualization purpose to know what is going on)

    Parameters:
    results_dict (dict): Dictionary containing the results with modified spectrograms.
    """
    # Select a random time stamp
    time_stamp_idx = random.randint(0, len(next(iter(results_dict['spectrograms'].values()))) - 1)
    
    # Plot each spectrogram for the selected time stamp
    plt.figure(figsize=(15, 20))
    
    for i, (key, entries) in enumerate(results_dict['spectrograms'].items()):
        plt.subplot(5, 5, i + 1)  # Assuming there are 25 spectrograms to plot (5x5 grid)
        entry = entries[time_stamp_idx]
        librosa.display.specshow(entry['S_db_gaussian'], sr=results_dict['sampling_rate'], x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'{key}')
    
    plt.tight_layout()
    plt.show()



# %% [markdown]
# So the approach here is to do the transforms in the __init__ function only and not already in the __getitem__ function.
#
# Takeaways:
#
# - All of the features reside in memory when extracting them with the __init__ function, and are easily retrievable afterwards. However in the beginning, there is A LOT of memory needed and most probably not scalable. On the contrary, if we would extract them in the __getitem__ function, it would take much longer for each access of the dataset

# %% [markdown]
# ### Hyperparameters
# DON'T change the Immutables!!!

# %%
# Immutables: Only here for readability of code
SAMPLING_RATE = 44100

SW_SIZE = 1
SW_HOP = 0.5
TARGET_BATCH_SIZE = 500


# %%
class BirdDataset(torch.utils.data.Dataset):
    def __init__(self, labels_file, song_dir, sw_size=1, sw_hop=0.5):
        """
        bird_labels: CSV file of the labels
        song_dir: directory to the bird song files
        song_files: a list of filenames in the song_dir
        sw_size: sliding window size in seconds
        sw_hop: sliding window hop in seconds
        song_chunks: transformed chunks for every song_file
        """
        self.bird_labels = pd.read_csv(labels_file)
        
        self.song_dir = song_dir
        self.song_files = [x for x in os.listdir(song_dir) if x[:2] == "XC"] # only take the labels we can match with an entry in labels.csv
        
        self.sw_size = sw_size
        self.sw_hop = sw_hop
        
        self.song_chunks = [self.getFeatures(os.path.join(song_dir, x), SAMPLING_RATE) for x in self.song_files] # Extracts all of the features of the song files


    # We measure the length of the dataset by the number of training points (and not by the size of the label csv)
    def __len__(self):
        return len(self.song_files)

    # Adjusted transforms by Jan and Kai with a overlapping sliding window
    # NOTE: all the bottleneck parts have been commented out for simplicity
    """
        I have commented out all the CONVOLUTIONS and the GAUSSIAN TRANSFORMATIONS
        --> It should be straightforward to uncomment them again
    """
    def getFeatures(self, fname, target_sr):

        """
            S: sampling rate
            S_t: transformed sampling
            
        """
        
        y, sr = librosa.load(fname, sr=target_sr)
        
        # Calculate no. of samples from sliding/hop window size
        window_samples = int(self.sw_size * sr)
        hop_samples = int(self.sw_hop * sr)
        
        # No. of windows that fit in the entire audio file
        num_windows = 1 + (len(y) - window_samples) // hop_samples
        
        # Preallocate and fill an array with chunks
        windows = np.zeros((num_windows, window_samples))
        for i in range(num_windows):
            start = i * hop_samples
            end = start + window_samples
            windows[i, :] = y[start:end]

        chunks = []
        for i, w in enumerate(windows):

            if len(w) < window_samples:
                break # Detect last sample
            
            # Short-time Fourier Transform
            S = librosa.stft(w) # (S,) --> (1025, S_t)
            S_db = librosa.amplitude_to_db(np.abs(S)) # Adjust spectrogram to dB scale
            
            """
                Bottleneck COMMENTED OUT: Gaussian filters to spectrograms
            """
            #x_means = np.linspace(0, S_db.shape[1], 5)
            #y_means = np.linspace(0, S_db.shape[0], 5)
            #std_dev_x = S_db.shape[1] / 10
            #std_dev_y = S_db.shape[0] / 10
            
            # Apply Gaussian masks to the spectrogram
            #gaussian_spectrograms = apply_gaussian_to_spectrogram(S_db, x_means, y_means, std_dev_x, std_dev_y)
            #output_vectors = []
            # M: masks
            """
                Bottleneck COMMENTED OUT: Convolutions
            """
            #for key, data in gaussian_spectrograms.items():
                #S_db_gaussian = data['S_db_gaussian'] # 2D spectrogram
                # O: channels
                #output_vector_np = apply_conv2d(S_db_gaussian) # O
                #output_vectors.append(output_vector_np.flatten())
                #output_vector_np = S_db_gaussian
            output_vectors = np.array(gaussian_spectrograms)
            chunks.append(output_vectors)
        chunks = np.array(chunks)

        return chunks

    # Basically the equivalent of implementing dataset[idx]
    def __getitem__(self, idx):
        song_fname = self.song_files[idx]

        """
            COMMENTED OUT: Here I have uncommented the previous approach to get the chunks in the getitem function only
            This has moved to the __init__, and we can simply access the chunks by the self.song_chunks array
        """
        #chunks = self.getFeatures(os.path.join(self.song_dir, song_fname))
        chunks = self.song_chunks[idx]

        
        label_df = self.bird_labels[self.bird_labels["XC_id"] == song_fname[:8]] # Matching XC identifier contained in filename with labels.csv entries

        assert not label_df.empty, "No matching labels for {}".format(song_fname) # Important right?
        
        # Edge Case: Choosing the first entry if we get 2 entries (bc sometimes the same filename is to be found in the gr. manaus and amazonas folders)
        label = (label_df["bid"].values)[0]
        
        return chunks, label

# %%
"""
    This object allows us to sample dynamically sized batches from our training set
    --> We need this because we have dramatically different lengths of sound files and padding would be too inefficent
"""
class DynamicBatchSampler(Sampler):
    def __init__(self, dataset, target_batch_size):
        self.dataset = dataset
        self.target_batch_size = target_batch_size
        self.indices = list(range(len(dataset)))
        self.indices.sort(key=lambda x: dataset[x][0].shape[0], reverse=True)  # Here we sort by length, as to help the Sampler group similarly sized elements together

    """
        Complicated code I know: Basically we are trying to squeeze in as many collection of chunks
        (which correspond to an audio file) into a batch, and if the next collection doesn't fit in to
        the current batch, we create a new batch
    """
    def __iter__(self):
        batch = []
        current_batch_size = 0
        for idx in self.indices:
            item_size = self.dataset[idx][0].shape[0]
            if current_batch_size + item_size > self.target_batch_size and batch:
                yield batch
                batch = []
                current_batch_size = 0
            batch.append(idx)
            current_batch_size += item_size
        if batch:
            yield batch

    # The total amount of batches we will get with our data
    def __len__(self):
        return math.ceil(sum(self.dataset[idx].shape[0] for idx in self.indices) / self.target_batch_size)


# %%
def collate_fn(batch):
    
    audio_data = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    # - Is batch item a torch.Tensor? --> if not convert it to one
    # Pad the rest of the sequence
    audio_data_padded = pad_sequence([ad if isinstance(ad, torch.Tensor) else torch.tensor(ad) for ad in audio_data],
                                     batch_first=True, padding_value=0.0)

    # Are labels torch.Tensors? --> if not convert them to torch.Tensor's
    if all(isinstance(label, torch.Tensor) for label in labels):
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=0)  # 0 as padding value, because we have NaN:0
    else:
        labels_padded = torch.tensor([label for label in labels], dtype=torch.long)

    return audio_data_padded, labels_padded


# %%
b = BirdDataset("./data/labels.csv", "./data/songs_condensed")
batch_sampler = DynamicBatchSampler(b, target_batch_size=TARGET_BATCH_SIZE)

# %%
loader = DataLoader(b, batch_sampler=batch_sampler, collate_fn=collate_fn)

# Usage in training
for x_train, x_label in loader:
    # Training logic here
    pass

# %% [markdown]
# ### Statistics - Performance Bottlenecks

# %% [markdown]
# Loading 13 samples with convolutions and gaussian transforms: 70 sec.
# Loading 13 samples without convolutions and without gauss. transorms: 4 sec.
