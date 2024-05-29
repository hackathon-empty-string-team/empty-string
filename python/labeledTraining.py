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

from tqdm import tqdm

#import nn
import torch.nn as nn


# for PCA plotting
from sklearn.decomposition import PCA


    
    
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
    
def preprocess_chunck(y_window, lower_bound_freq=800, higher_bound_freq=9000, target_sr=16000, n_mels=128, hop_length=2048, n_fft=4096):
    
    
    # Compute STFT
    S = librosa.feature.melspectrogram(y=y_window, sr=target_sr, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft)
    S_db = librosa.amplitude_to_db(np.abs(S))
    
    # Define Gaussian parameters
    #x_means = np.linspace(0, S_db.shape[1], 3)
    y_means = higher_bound_freq-lower_bound_freq / 2
    #std_dev_x = S_db.shape[1] / 3
    std_dev_y = higher_bound_freq-lower_bound_freq / 3
    
    # Apply Gaussian masks to the spectrogram
    #gaussian_spectrograms = apply_gaussian_to_spectrogram(S_db, x_means, y_means, std_dev_x, std_dev_y)
    #gaussian_spectrograms = apply_gaussian_to_spectrogram_1d(S_db, y_means, std_dev_y)
    gaussian_spectrograms=S_db
    

    #return gaussian_spectrograms['y_0']["S_db_gaussian"]
    return gaussian_spectrograms




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
SAMPLING_RATE = 48000 #xenocanto

SW_SIZE = 5
SW_HOP = 2.5
TARGET_BATCH_SIZE = 500


class BirdDataset(torch.utils.data.Dataset):
    def __init__(self, labels_file, song_dir, sw_size=5, sw_hop=2.5):
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
        features = [self.getFeatures(os.path.join(song_dir, x), SAMPLING_RATE, count_only=True) for x in self.song_files]
        self.song_chunk_counts = [x[1] for x in features]
        self.startingIndices = self.getStartingIndices(self.song_chunk_counts)
        self.labels = self.bird_labels['bid'].tolist() 

        self.class_indices = self.get_class_indices()

    def get_class_indices(self):
        class_indices = {}
        for idx, label in enumerate(self.labels):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)
        return class_indices



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
    def getFeatures(self, fname, target_sr, lower_bound_freq=2000, higher_bound_freq=7000, count_only=False):

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

        

        if count_only:
            return None, num_windows
        
        # Preallocate and fill an array with chunks
        if num_windows <= 1:
            # If the audio file is too short to have 2 chunks, pad it with its own data
            num_windows = 2
            needed_size=window_samples+hop_samples
            y = np.pad(y, (0, needed_size), mode='wrap')
        
            
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
        local_chunk_idx = chunk_idx - int(self.startingIndices[song_idx]) #it is the index of the chunk within the song
        
        # Load only the required chunk
        chunks, _ = self.getFeatures(os.path.join(self.song_dir, song_fname), target_sr=SAMPLING_RATE)  # Could be optimized to load only the required chunk
        #print(chunks.shape)
        chunk = chunks[local_chunk_idx]

        #normalize the chunk
        chunk = (chunk - np.mean(chunk)) / np.std(chunk)
        
        return chunk, self.getLabel(song_fname)



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
    triplets: tensor of triplets of the form (anchor_idx, positive_idx, negative_idx)
    """
    triplets = []
    used_indices = set()
    num_triplets = 0
    num_samples = outputs.size(0)
    
    
    for i in range(num_samples):
        if i in used_indices:
            continue
        
        anchor_label = labels[i]
        
        # Get indices of all samples with the same label, excluding the anchor
        positive_indices = (labels == anchor_label).nonzero().squeeze(1).tolist()
        positive_indices = [idx for idx in positive_indices if idx != i and idx not in used_indices]
        
        # Get indices of all samples with different label
        negative_indices = (labels != anchor_label).nonzero().squeeze(1).tolist()
        negative_indices = [idx for idx in negative_indices if idx not in used_indices]
        
        if not positive_indices or not negative_indices:
            continue
        
        # Select a random positive and negative sample
        positive_idx = random.choice(positive_indices)
        negative_idx = random.choice(negative_indices)
        
        triplets.append((i, positive_idx, negative_idx))
        used_indices.update({i, positive_idx, negative_idx})
        num_triplets += 1
        
        if len(used_indices) >= num_samples:
            break
    print(num_triplets)
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
    
class MyModelVGG(nn.Module):
    def __init__(self, numChannels):
        # call the parent constructor
        super(MyModelVGG, self).__init__()
        # initialize the CONV layers
        self.conv1 = nn.Conv2d(in_channels=numChannels, out_channels=128,
            kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64,
            kernel_size=(3, 3))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32,
            kernel_size=(3, 3))
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=16,
            kernel_size=(3, 3))
        # initialize the linear layers
        self.fc1 = nn.Linear(480, 128)

    def forward(self, x):
        # pass the input through the first set of CONV => RELU =>
        # example if x is of shape (batch_size, 1, 1025, 129)
        # POOL layers
        #print(x.shape)
        x = nn.ReLU()(self.conv1(x)) # shape (batch_size, 128, 1023, 127)
        x = nn.MaxPool2d(kernel_size=(2, 2), stride=2)(x) # shape (batch_size, 128, 511, 63)
        # pass the input through the second set of CONV => RELU =>
        # POOL layers
        #print(x.shape)
        x = nn.ReLU()(self.conv2(x)) # shape (batch_size, 64, 509, 61)
        x = nn.MaxPool2d(kernel_size=(2, 2), stride=2)(x) # shape (batch_size, 64, 254, 30)
        # pass the input through the third set of CONV => RELU =>
        # POOL layers
        x = nn.ReLU()(self.conv3(x)) # shape (batch_size, 32, 252, 28)
        x = nn.MaxPool2d(kernel_size=(2, 2), stride=2)(x) # shape (batch_size, 32, 126, 14)
        # pass the input through the fourth set of CONV => RELU =>
        # POOL layers
        x = nn.ReLU()(self.conv4(x)) # shape (batch_size, 16, 124, 12)
        x = nn.MaxPool2d(kernel_size=(2, 2), stride=2)(x) # shape (batch_size, 16, 62, 6)
        # flatten the volume, then pass it through a fully-connected layer
        x = nn.Flatten()(x) # shape (batch_size, 5952)
        x = nn.ReLU()(self.fc1(x)) # shape (batch_size, 128)
        # return the feature map
        return x
    
class BalancedBatchSampler(Sampler):
    def __init__(self, dataset):
        self.dataset = dataset
        self.class_indices = dataset.class_indices
        self.classes = list(self.class_indices.keys())
        self.num_samples_per_class = min(len(indices) for indices in self.class_indices.values())

    def __iter__(self):
        # Goal: Iterate over the dataset in a balanced way, i.e., each batch will contain the same number of samples per class
        balanced_indices = []
        for class_label in self.classes: # Iterate over all classes
            indices = np.random.choice(self.class_indices[class_label], self.num_samples_per_class, replace=False) # Choose random samples without replacement
            balanced_indices.extend(indices) # Add the indices to the list of balanced indices
        np.random.shuffle(balanced_indices) # Shuffle the indices to ensure that the classes are mixed
        return iter(balanced_indices)

    def __len__(self):
        return self.num_samples_per_class * len(self.classes)
    
def plot_vectors_with_pca(vectors, labels):
    # Perform PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(vectors)
    
    
    pca_result=np.array(pca_result)
    pca_result=pca_result[np.array(labels)<100]
    labels=np.array(labels)[np.array(labels)<100]



    plt.figure(figsize=(8, 6))
    for label in np.unique(labels):
        indices = np.where(labels == label)
        plt.scatter(pca_result[indices, 0], pca_result[indices, 1], label=label)
    plt.title('PCA Visualization of Model Output Vectors')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_vectors_with_pca_multiple(vectors, labels):
    # Perform PCA with 4 dimensions and plot them in 2D
    pca = PCA(n_components=4)
    pca_result = pca.fit_transform(vectors)
    pca_result=np.array(pca_result)
    pca_result=pca_result[np.array(labels)<100]
    labels=np.array(labels)[np.array(labels)<100]
    # Plot multiple 2D projections
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs_i = 0
    axs_j = 0
    for i, j in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]:
        
        for label in np.unique(labels):
            indices = np.where(labels == label)
            axs[axs_i, axs_j].scatter(pca_result[indices, i], pca_result[indices, j], label=label)
        axs[axs_i, axs_j].set_xlabel(f'Principal Component {i + 1}')
        axs[axs_i, axs_j].set_ylabel(f'Principal Component {j + 1}')
        axs[axs_i, axs_j].legend()
        axs[axs_i, axs_j].grid(True)
        axs_i += 1
        if axs_i == 2:
            axs_i = 0
            axs_j += 1
    plt.suptitle('PCA Visualization of Model Output Vectors')
    plt.tight_layout()

    plt.show()


# %%
#b = BirdDataset("../data/labels.csv", "../data/songs_condensed")
b_sub= BirdDataset("../data/labels.csv", "../data/sub_set")

# %%
#balanced_sampler = BalancedBatchSampler(b_sub)
#loader_bal = DataLoader(b_sub, batch_size=100, sampler=balanced_sampler)
loader = DataLoader(b_sub, batch_size=100, shuffle=True)

loss_fn=torch.nn.TripletMarginLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model=MyConv2D(1).to("cuda")
model=MyModelVGG(1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

nb_epochs = 3
model.train()

for epoch in range(nb_epochs):
    running_loss = 0.0
    
    # Initialize the progress bar for the current epoch
    progress_bar = tqdm(enumerate(loader, 0), total=len(loader), desc=f"Epoch {epoch + 1}/{nb_epochs}", unit="batch")
    
    for i, data in progress_bar:
        inputs, labels = data
        inputs = inputs.float().to(device)
        labels = labels.to(device)
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

        # Update the progress bar description with the current loss
        if i % 3 == 2:
            progress_bar.set_postfix(loss=running_loss / 200)
            running_loss = 0.0
    
    print(f"Epoch {epoch + 1} finished")
    torch.save(model.state_dict(), "model_sub.pth")







#create feature vectors one batch of sound
def create_feature_vectors(model, loader):
    feature_vectors = []
    labels = []
    
    for i, data in enumerate(loader):
        inputs, labels_batch = data
        inputs = inputs.float().to(device)
        labels_batch = labels_batch.to(device)
        outputs = model(inputs.unsqueeze(1))
        feature_vectors.extend(outputs.detach().cpu().numpy())
        labels.extend(labels_batch.detach().cpu().numpy())
    
    return feature_vectors, labels

# %%
# Load the model
model = MyModelVGG(1).to(device)
model.load_state_dict(torch.load("model_sub.pth"))
model.eval()

# Create a DataLoader for the entire dataset
loader = DataLoader(b_sub, batch_size=100, shuffle=True)

# Create feature vectors for all the sounds in the dataset
feature_vectors, labels = create_feature_vectors(model, loader)


#plot_vectors_with_pca(feature_vectors, labels)
plot_vectors_with_pca_multiple(feature_vectors, labels)
