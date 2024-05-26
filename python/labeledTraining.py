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
#import tensorflow as tf
import random
import torch
import time
from torch.utils.data import Sampler, Dataset, DataLoader
import math
from torch.nn.utils.rnn import pad_sequence

#import nn
import torch.nn as nn

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

    
    
def gaussian_1d(x, y, mean_y, std_dev_y):
    """
    Generates a 1D Gaussian mask only on frequency axis.

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
    gauss_y = np.exp(-0.5 * ((y - mean_y) / std_dev_y) ** 2)
    gauss_x = np.ones(x.shape)
    return gauss_x * gauss_y


    
def apply_gaussian_to_spectrogram_1d(S_db, y_means, std_dev_y):
    """
    Applies Gaussian masks to a spectrogram and returns the results.


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
    
   
    gaussian_mask = gaussian_1d(x, y_grid, y_means, std_dev_y)
    S_db_gaussian = S_db * gaussian_mask
    key = f'y_0'
    results[key] = {
        'S_db_gaussian': S_db_gaussian,
        'y_mean': y_means,
        'std_dev_y': std_dev_y
    }
    
    return results
    
def preprocess_chunck(y_window, lower_bound_freq=800, higher_bound_freq=9000):
    
    
    # Compute STFT
    S = librosa.stft(y_window)
    S_db = librosa.amplitude_to_db(np.abs(S))
    
    # Define Gaussian parameters
    #x_means = np.linspace(0, S_db.shape[1], 3)
    y_means = higher_bound_freq-lower_bound_freq / 2
    #std_dev_x = S_db.shape[1] / 3
    std_dev_y = higher_bound_freq-lower_bound_freq / 3
    
    # Apply Gaussian masks to the spectrogram
    #gaussian_spectrograms = apply_gaussian_to_spectrogram(S_db, x_means, y_means, std_dev_x, std_dev_y)
    gaussian_spectrograms = apply_gaussian_to_spectrogram_1d(S_db, y_means, std_dev_y)
    

    return gaussian_spectrograms['y_0']["S_db_gaussian"]

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

SW_SIZE = 5
SW_HOP = 2.5
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
        
        # Initialize starting indices and chunk counts per song file
        features = [self.getFeatures(os.path.join(song_dir, x), SAMPLING_RATE) for x in self.song_files]
        self.song_chunk_counts = [x[1] for x in features]
        self.startingIndices = self.getStartingIndices(self.song_chunk_counts)


    def getStartingIndices(self, chunk_nums):
        indices = np.zeros(len(chunk_nums) + 1)
        count = 0
        for i, chunk_num in enumerate(chunk_nums):
            indices[i] = count
            count += chunk_num
        indices[-1] = count  # Set the last index to the total count. for example, if we have 3 songs with 2, 3, and 4 chunks, the starting indices will be [0, 2, 5, 9]
        return indices

    def getSongIdx(self, chunk_idx):

        indices = self.startingIndices
        for i, idx in enumerate(indices):

            if idx > chunk_idx:

                return (i - 1)
                
        return -1
            
    
    # We measure the length of the dataset by the number of training points (and not by the size of the label csv)
    def __len__(self):
        return int(self.startingIndices[-1])

    # Adjusted transforms by Jan and Kai with a overlapping sliding window
    # NOTE: all the bottleneck parts have been commented out for simplicity
    """
        I have commented out all the CONVOLUTIONS and the GAUSSIAN TRANSFORMATIONS
        --> It should be straightforward to uncomment them again
    """
    def getFeatures(self, fname, target_sr, lower_bound_freq=2000, higher_bound_freq=7000):

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
            
            output_vectors = preprocess_chunck(w, lower_bound_freq=lower_bound_freq, higher_bound_freq=higher_bound_freq)
            chunks.append(output_vectors)
        chunks = np.array(chunks)
        return chunks, num_windows

    def getLabel(self, song_fname):

        label_df = self.bird_labels[self.bird_labels["XC_id"] == song_fname[:8]] # Matching XC identifier contained in filename with labels.csv entries

        assert not label_df.empty, "No matching labels for {}".format(song_fname) # Important right?
        
        # Edge Case: Choosing the first entry if we get 2 entries (bc sometimes the same filename is to be found in the gr. manaus and amazonas folders)
        return (label_df["bid"].values)[0]

    # Basically the equivalent of implementing dataset[idx]
    def __getitem__(self, chunk_idx):

        song_idx = self.getSongIdx(chunk_idx)
        song_fname = self.song_files[song_idx]

        # Calculate the local chunk index within the song
        local_chunk_idx = chunk_idx - int(self.startingIndices[song_idx])

        # Load only the required chunk
        chunks, _ = self.getFeatures(os.path.join(self.song_dir, song_fname), target_sr=SAMPLING_RATE)
        chunk = chunks[local_chunk_idx]
        
        return chunk, self.getLabel(song_fname)


# %%
def getTriplet(ds, bid):
    
    df = ds.bird_labels
    
    pos_df = df[df["bid"] == bid]
    pos_df = pos_df[pos_df["Filename"].isin(ds.song_files)]["Filename"].values
    neg_df = df[df["bid"] != bid]
    neg_df = neg_df[neg_df["Filename"].isin(ds.song_files)]["Filename"].values
    # Choose random pos/neg file
    pos_fname = np.random.choice(pos_df)
    neg_fname = np.random.choice(neg_df)

    # Get index in the song_files list
    pos_findex = ds.song_files.index(pos_fname)
    neg_findex = ds.song_files.index(neg_fname)

    print(ds.startingIndices)

    # Get random chunk from the random file
    pos_cindex = random.randint(ds.startingIndices[pos_findex], ds.startingIndices[pos_findex + 1])
    neg_cindex = random.randint(ds.startingIndices[neg_findex], ds.startingIndices[neg_findex + 1])
    return pos_cindex, neg_cindex


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
    
def create_triplets(outputs, labels):
    """
    Create triplets from the output of the model and the labels
    
    Parameters:
    outputs: tensor of shape (batch_size, feature size)
    labels: tensor of shape (batch_size,)
    
    Returns:
    triplets: list of tuples of the form (anchor_idx, positive_idx, negative_idx)
    """
    triplets = []
    num_triplets = 0
    
    for i in range(outputs.size(0)):
        anchor = outputs[i]
        anchor_label = labels[i]
        
        # Get indices of all samples with the same label, exc
        positive_indices = (labels == anchor_label).nonzero().squeeze(1)
        
        # Get indices of all samples with different label
        negative_indices = (labels != anchor_label).nonzero().squeeze(1)
        
        if positive_indices.size(0) == 0 or negative_indices.size(0) == 0:
            continue
        
        # Select a random positive and negative sample
        positive_idx = random.choice(positive_indices)
        negative_idx = random.choice(negative_indices)
        
        triplets.append((i, positive_idx, negative_idx))
        num_triplets += 1
    
    return torch.tensor(triplets)

class MyConv2D(nn.Module):
    def __init__(self, numChannels):
        # call the parent constructor
        super(MyConv2D, self).__init__()
        # initialize first CONV layer
        self.conv1 = nn.Conv2d(in_channels=numChannels, out_channels=128,
            kernel_size=(3, 3))
        
    def forward(self, x):
        # pass the input through the first CONV layer and then through
        # the ReLU activation function followed by a max-pooling layer
        print(x.shape)
        x = self.conv1(x)
        x = nn.AvgPool2d(kernel_size=(x.shape[2], x.shape[3]), stride=2)(x)
        x=x.squeeze(-1).squeeze(-1)
        
        
        # return the feature map
        return x


# %%
b = BirdDataset("../data/labels.csv", "../data/songs_condensed")

# %%
loader = DataLoader(b, batch_size=10, shuffle=True)

loss_fn=torch.nn.TripletMarginLoss()
model=MyConv2D(1).to("cuda")
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

nb_epochs = 3
model.train()

for epoch in range(nb_epochs):
    running_loss = 0.0
    for i, data in enumerate(loader, 0):
        inputs, labels = data
        inputs=inputs.float().to("cuda")
        labels=labels.to("cuda")
        optimizer.zero_grad()
        outputs = model(inputs.unsqueeze(1))
        triplets = create_triplets(outputs, labels)
        if len(triplets) == 0:
            continue
        anchor = outputs[triplets[:, 0]]
        positive = outputs[triplets[:, 1]]
        negative = outputs[triplets[:, 2]]
        loss = loss_fn(anchor, positive, negative)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if i % 20 == 19:

            print(f"Epoch {epoch + 1}, iteration {i + 1}: loss {running_loss / 200}")
            running_loss = 0.0
    
    print(f"Epoch {epoch + 1} finished")
