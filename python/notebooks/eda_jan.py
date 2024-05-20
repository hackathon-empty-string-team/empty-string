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
## cockpit for directories
# Directory containing the audio files
# audio_dir = "data/soundscape_data"
audio_dir = (
    "../data/SoundMeters_Ingles_Primary-20240519T132658Z-002/SoundMeters_Ingles_Primary"
)

# Directory to save features
features_dir = "../data/features"
os.makedirs(features_dir, exist_ok=True)

# Directory to save clusters information
clusters_dir = "../data/clusters"
os.makedirs(clusters_dir, exist_ok=True)

# %%
# Parameters for windowing
window_size = 10  # window size in seconds
hop_size = 10  # hop size in seconds

# Define frequency bands (in Hz)
bands = {
    "Sub-bass": (20, 60),
    "Bass": (60, 250),
    "Low Midrange": (250, 500),
    "Midrange": (500, 2000),
    "Upper Midrange": (2000, 4000),
    "Presence": (4000, 6000),
    "Brilliance": (6000, 20000),
}

# Iterate over each audio file in the directory
for filename in os.listdir(audio_dir):
    if filename.endswith(".wav"):
        file_path = os.path.join(audio_dir, filename)
        y, sr = librosa.load(file_path, sr=None)

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
        feature_file = os.path.join(
            features_dir, f"{os.path.splitext(filename)[0]}_features.npy"
        )
        joblib.dump((all_features, scaler), feature_file)


# %%
# Number of clusters
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

# Save clustering results
clustering_results = {"clusters": clusters, "kmeans": kmeans, "pca": pca}
joblib.dump(clustering_results, os.path.join(clusters_dir, "clustering_results.pkl"))


# %%
# Plot the PCA-reduced features with cluster labels using a legend
plt.figure(figsize=(10, 6))

# Define a colormap
colors = plt.cm.tab10(np.arange(kmeans.n_clusters))

for cluster_label in np.unique(clusters):
    cluster_points = features_pca[clusters == cluster_label]
    plt.scatter(
        cluster_points[:, 0],
        cluster_points[:, 1],
        s=50,
        color=colors[cluster_label],
        label=f"Cluster {cluster_label}",
    )

plt.title("PCA of Clustered Frequency Band Features")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.show()


# %%
# Plot the clusters
plt.figure(figsize=(10, 6))
for i in range(n_clusters):
    plt.plot(all_features[clusters == i].mean(axis=0), label=f"Cluster {i}")
plt.legend()
plt.title("Clustered Frequency Band Features")
plt.xlabel("Feature Index (Frequency Bands)")
plt.ylabel("Mean Feature Value (Energy in dB)")
plt.show()


# %%
# Function to plot the spectrogram
def plot_spectrogram(y, sr, title):
    S = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="log")
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.show()


# %%
# Parameters for windowing
window_size = 10  # window size in seconds
hop_size = 10  # hop size in seconds

# Load clustering results
clustering_results = joblib.load(os.path.join(clusters_dir, "clustering_results.pkl"))
clusters = clustering_results["clusters"]

# Load all features
all_features = []
audio_segments = []

for feature_file in os.listdir(features_dir):
    if feature_file.endswith("_features.npy"):
        features, scaler = joblib.load(os.path.join(features_dir, feature_file))
        filename = feature_file.replace("_features.npy", ".wav")
        file_path = os.path.join(audio_dir, filename)
        y, sr = librosa.load(file_path, sr=None)

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


# %%
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

        # Plot the spectrogram
        plot_spectrogram(
            y_representative, sr, f"Spectrogram for Cluster {cluster_label}"
        )

    except Exception as e:
        print(f"Could not play audio for cluster {cluster_label}: {e}")

# %%
scaler = StandardScaler()
all_features_scaled = scaler.fit_transform(all_features)
# Fit PCA
pca = PCA().fit(all_features_scaled)

# Method 1: Variance Explained
explained_variance = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance)

# Plot the cumulative explained variance
plt.figure(figsize=(10, 6))
plt.plot(cumulative_explained_variance, marker="o")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Explained Variance vs. Number of Principal Components")
plt.grid(True)
plt.show()


# %%
# Method 2: Scree Plot
plt.figure(figsize=(10, 6))
plt.plot(np.arange(1, len(explained_variance) + 1), explained_variance, marker="o")
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance")
plt.title("Scree Plot")
plt.grid(True)
plt.show()


# %%
# Method 3: Kaiser Criterion
eigenvalues = pca.explained_variance_
kaiser_criterion = np.sum(eigenvalues > 1)


# IMO this doesnt make sense at the moment, we need to extract more features
print(f"Number of components selected by Kaiser Criterion: {kaiser_criterion}")


# %%

# %%
