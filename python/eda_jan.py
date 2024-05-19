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
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from IPython.display import Audio, display 

# %%
# Load the entire audio file
cwd = os.getcwd()
relative_path = "data/soundscape_data/PER_001_S01_20190116_100007Z.flac"
file_path = os.path.join(cwd, relative_path)
y, sr = librosa.load(file_path, sr=44100)

# %%
# split soundfile in to 10s chunks
window_size = 10  # window size in seconds
hop_size = 10     # hop size in seconds

# Convert window and hop size to samples
window_samples = int(window_size * sr)
hop_samples = int(hop_size * sr)

# Total number of windows
num_windows = (len(y) - window_samples) // hop_samples + 1

print(f"Total number of windows: {num_windows}")


# %%
# Define frequency bands (in Hz)
bands = {
    'Sub-bass': (20, 60),
    'Bass': (60, 250),
    'Low Midrange': (250, 500),
    'Midrange': (500, 2000),
    'Upper Midrange': (2000, 4000),
    'Presence': (4000, 6000),
    'Brilliance': (6000, 20000)
}

# Initialize a list to hold the features
all_features = []

for i in range(num_windows):
    start_sample = i * hop_samples
    end_sample = start_sample + window_samples
    y_window = y[start_sample:end_sample]

    # Compute STFT
    S = librosa.stft(y_window)
    S_db = librosa.amplitude_to_db(np.abs(S))
    
    # Compute features for each band
    features = []
    for band, (low_freq, high_freq) in bands.items():
        low_bin = int(np.floor(low_freq * (S.shape[0] / sr)))
        high_bin = int(np.ceil(high_freq * (S.shape[0] / sr)))
        band_energy = np.mean(S_db[low_bin:high_bin, :], axis=0)
        features.append(band_energy)
    
    # Flatten the feature array and add to all_features
    features_flat = np.concatenate(features)
    all_features.append(features_flat)

# Convert to numpy array
all_features = np.array(all_features)


# %%
# Reduce dimensionality with PCA
pca = PCA(n_components=2)
features_reduced = pca.fit_transform(all_features)

# Perform k-means clustering
kmeans = KMeans(n_clusters=5)  # Example: 5 clusters
clusters = kmeans.fit_predict(features_reduced)

# Plot the clusters
plt.figure(figsize=(10, 6))
scatter = plt.scatter(features_reduced[:, 0], features_reduced[:, 1], c=clusters, cmap='viridis')
plt.title('Clustered Frequency Band Features')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(scatter, label='Cluster')
plt.show()


# %%
# Play the audio for a representative sample from each cluster
for cluster_label in np.unique(clusters):
    # Find the first data point in the cluster
    representative_index = np.where(clusters == cluster_label)[0][0]
    
    # Use the original audio window at the representative index
    start_sample = representative_index * hop_samples
    end_sample = start_sample + window_samples
    y_representative = y[start_sample:end_sample]
    
    print(f"Cluster {cluster_label} representative audio:")
    display(Audio(data=y_representative, rate=sr))


# %% [markdown]
# ## pipeline for all the files

# %%
import os
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import librosa
from IPython.display import Audio, display
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier



# %%
# Directory containing the audio files
# audio_dir = "data/soundscape_data"
audio_dir = "data/SoundMeters_Ingles_Primary-20240519T132658Z-002/SoundMeters_Ingles_Primary"

# Parameters for windowing
window_size = 10  # window size in seconds
hop_size = 10     # hop size in seconds

# Define frequency bands (in Hz)
bands = {
    'Sub-bass': (20, 60),
    'Bass': (60, 250),
    'Low Midrange': (250, 500),
    'Midrange': (500, 2000),
    'Upper Midrange': (2000, 4000),
    'Presence': (4000, 6000),
    'Brilliance': (6000, 20000)
}

# Directory to save features
features_dir = "features"
os.makedirs(features_dir, exist_ok=True)

# Iterate over each audio file in the directory
for filename in os.listdir(audio_dir):
    if filename.endswith(".wav"):
        file_path = os.path.join(audio_dir, filename)
        y, sr = librosa.load(file_path, sr=44100)
        
        # Convert window and hop size to samples
        window_samples = int(window_size * sr)
        hop_samples = int(hop_size * sr)

        # Total number of windows in the current file
        num_windows = (len(y) - window_samples) // hop_samples + 1

        all_features = []

        for i in range(num_windows):
            start_sample = i * hop_samples
            end_sample = start_sample + window_samples
            y_window = y[start_sample:end_sample]

            # Compute STFT
            S = librosa.stft(y_window)
            S_db = librosa.amplitude_to_db(np.abs(S))

            # Compute features for each band
            features = []
            for band, (low_freq, high_freq) in bands.items():
                low_bin = int(np.floor(low_freq * (S.shape[0] / sr)))
                high_bin = int(np.ceil(high_freq * (S.shape[0] / sr)))
                band_energy = np.mean(S_db[low_bin:high_bin, :], axis=0)
                features.append(band_energy)

            # Flatten the feature array and add to all_features
            features_flat = np.concatenate(features)
            all_features.append(features_flat)

        # Convert to numpy array
        all_features = np.array(all_features)

        # Standardize features
        scaler = StandardScaler()
        all_features = scaler.fit_transform(all_features)

        # Save features to disk
        feature_file = os.path.join(features_dir, f"{os.path.splitext(filename)[0]}_features.npy")
        joblib.dump((all_features, scaler), feature_file)


# %%
# Directory to load features
features_dir = "features"
n_clusters = 5


# Load all features
all_features = []
for feature_file in os.listdir(features_dir):
    if feature_file.endswith("_features.npy"):
        features, _ = joblib.load(os.path.join(features_dir, feature_file))
        all_features.append(features)

# Combine all features into a single array
all_features = np.vstack(all_features)

# Perform PCA for 2D visualization
pca = PCA(n_components=2)
features_pca = pca.fit_transform(all_features)

# Perform k-means clustering
kmeans = KMeans(n_clusters=n_clusters)  # Example: 5 clusters
clusters = kmeans.fit_predict(all_features)

# Plot the PCA-reduced features with cluster labels
plt.figure(figsize=(10, 6))
scatter = plt.scatter(features_pca[:, 0], features_pca[:, 1], c=clusters, cmap='viridis')
plt.title('PCA of Clustered Frequency Band Features')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(scatter, label='Cluster')
plt.show()

# Save clustering results
clustering_results = {
    'clusters': clusters,
    'kmeans': kmeans,
    'pca': pca
}
joblib.dump(clustering_results, 'clustering_results.pkl')

# Plot the clusters
plt.figure(figsize=(10, 6))
for i in range(n_clusters):
    plt.plot(all_features[clusters == i].mean(axis=0), label=f'Cluster {i}')
plt.legend()
plt.title('Clustered Frequency Band Features')
plt.xlabel('Feature Index (Frequency Bands)')
plt.ylabel('Mean Feature Value (Energy in dB)')
plt.show()

# %%
# Directory containing the audio files
# audio_dir = "data/soundscape_data"
audio_dir = "data/SoundMeters_Ingles_Primary-20240519T132658Z-002/SoundMeters_Ingles_Primary"
# Directory to load features
features_dir = "features"

# Parameters for windowing
window_size = 10  # window size in seconds
hop_size = 10     # hop size in seconds

# Load clustering results
clustering_results = joblib.load('clustering_results.pkl')
clusters = clustering_results['clusters']

# Load all features
all_features = []
audio_segments = []

for feature_file in os.listdir(features_dir):
    if feature_file.endswith("_features.npy"):
        features, scaler = joblib.load(os.path.join(features_dir, feature_file))
        filename = feature_file.replace('_features.npy', '.wav')
        file_path = os.path.join(audio_dir, filename)
        y, sr = librosa.load(file_path, sr=44100)

        # Convert window and hop size to samples
        window_samples = int(window_size * sr)
        hop_samples = int(hop_size * sr)

        num_windows = (len(y) - window_samples) // hop_samples + 1
        for i in range(num_windows):
            start_sample = i * hop_samples
            end_sample = start_sample + window_samples
            y_window = y[start_sample:end_sample]
            audio_segments.append(y_window)
        all_features.append(features)

# Flatten the list of all features
all_features = np.vstack(all_features)

# Play the audio for a representative sample from each cluster
for cluster_label in np.unique(clusters):
    try:
        # Find the first data point in the cluster
        representative_index = np.where(clusters == cluster_label)[0][0]

        # Use the original audio segment at the representative index
        y_representative = audio_segments[representative_index]

        # Check if y_representative is not empty
        if y_representative.size == 0:
            raise ValueError("The audio segment is empty")

        print(f"Cluster {cluster_label} representative audio:")
        display(Audio(data=y_representative, rate=sr))

    except Exception as e:
        print(f"Could not play audio for cluster {cluster_label}: {e}")


# %%

# Fit PCA
pca = PCA().fit(all_features_scaled)

# Method 1: Variance Explained
explained_variance = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance)

# Plot the cumulative explained variance
plt.figure(figsize=(10, 6))
plt.plot(cumulative_explained_variance, marker='o')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance vs. Number of Principal Components')
plt.grid(True)
plt.show()



# %%
# Method 2: Scree Plot
plt.figure(figsize=(10, 6))
plt.plot(np.arange(1, len(explained_variance) + 1), explained_variance, marker='o')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance')
plt.title('Scree Plot')
plt.grid(True)
plt.show()



# %%
# Method 3: Kaiser Criterion
eigenvalues = pca.explained_variance_
kaiser_criterion = np.sum(eigenvalues > 1)


# IMO this doesnt make sense at the moment, we need to extract more features
print(f"Number of components selected by Kaiser Criterion: {kaiser_criterion}")



# %%
# Method 4: Cross-Validation
# Evaluate a classifier with different numbers of principal components

## do not run if you dont have time, this takes forever. 
# scores = []
# for n_components in range(1, len(explained_variance) + 1):
#     pca = PCA(n_components=n_components)
#     features_pca = pca.fit_transform(all_features_scaled)
#     classifier = RandomForestClassifier()  # Use your preferred model here
#     score = np.mean(cross_val_score(classifier, features_pca, clusters, cv=n_clusters))  # Assuming `clusters` are your labels
#     scores.append(score)

# # Plot cross-validation scores
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, len(explained_variance) + 1), scores, marker='o')
# plt.xlabel('Number of Principal Components')
# plt.ylabel('Cross-Validation Score')
# plt.title('Cross-Validation Score vs. Number of Principal Components')
# plt.grid(True)
# plt.show()

# # Choosing the number of components that explain at least 95% of the variance
# n_components_variance = np.argmax(cumulative_explained_variance >= 0.95) + 1
# print(f"Number of components to retain 95% variance: {n_components_variance}")

# # Choose the optimal number of components based on your analysis
# optimal_n_components = n_components_variance  # or based on the scree plot, cross-validation, etc.
# print(f"Optimal number of components: {optimal_n_components}")

# # Perform PCA with the selected number of components
# pca = PCA(n_components=optimal_n_components)
# features_pca = pca.fit_transform(all_features_scaled)

# %%

# %%
