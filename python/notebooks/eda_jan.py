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
                band_energy = np.median(S_db[low_bin:high_bin, :], axis=0)
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

# %% [markdown]
# ## New approach for data engineering

# %%

# Parameters
window_size = 10  # window size in seconds

# Path to the audio file
audio_file_path = '/python/data/SoundMeters_Ingles_Primary-20240519T132658Z-017/SoundMeters_Ingles_Primary/SM4XPRIZE_20240409_235202.wav'

# Load audio file
y, sr = librosa.load(audio_file_path, sr=None)

# Convert window size to samples
window_samples = int(window_size * sr)

# Create a 10-second chunk
y_window = y[:window_samples]

# Compute STFT
S = librosa.stft(y_window)
S_db = librosa.amplitude_to_db(np.abs(S))

# Plot spectrogram
plt.figure(figsize=(10, 6))
librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log', cmap='coolwarm')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
plt.tight_layout()


# %%
S_db

# %%
S_db.shape

# %%
S_db[0][0]

# %%
# Find dimensions of the spectrogram
dimensions = S_db.shape
num_freq_bins, num_time_frames = dimensions

# Generate a normal sinus function
# Here, we'll create a sinus function that matches the length of the frequency bins
x = np.linspace(0, 2 * np.pi, num_freq_bins)
sinus_function = np.sin(x)

# Multiply each column of the spectrogram with the sinus function
S_db_mod = S_db * sinus_function[:, np.newaxis]

# %%
plt.plot(x, np.sin(x - np.pi))

# %%
# Plot the modified spectrogram
plt.figure(figsize=(10, 6))
librosa.display.specshow(S_db_mod, sr=sr, x_axis='time', y_axis='log', cmap='coolwarm')
plt.colorbar(format='%+2.0f dB')
plt.title('Modified Spectrogram with Sinus Function')
plt.tight_layout()

# %%
S_db_mod.shape

# %%
import tensorflow as tf
from tensorflow import keras

# %%
# Reshape to add batch dimension (1, height, width, channels)
input_tensor = tf.reshape(S_db_mod, (1, S_db_mod.shape[0], S_db_mod.shape[1], 1))

# Create a Conv2D layer with 64 filters, 3x3 kernel size, and 'same' padding
conv_layer = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')

# Apply the Conv2D layer to the input tensor
output_tensor = conv_layer(input_tensor)

# To get the output as a vector with 64 features, you can apply global average pooling
output_vector = tf.keras.layers.GlobalAveragePooling2D()(output_tensor)

# Convert the output vector to a numpy array
output_vector_np = output_vector.numpy()

print("Output Vector Shape:", output_vector_np.shape)
print("Output Vector:", output_vector_np)

# %%
plt.hist(output_vector_np)
plt.show()

# %%
# Define the Sequential model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(28, 28, 1))
])

# Apply the model to the input tensor
feature_maps = model(input_tensor)

# Get the feature map with the highest average activation
average_activations = tf.reduce_mean(feature_maps, axis=(1, 2))
most_important_feature_index = tf.argmax(average_activations, axis=1).numpy()[0]

# Extract the most important feature map
most_important_feature_map = feature_maps[0, :, :, most_important_feature_index].numpy()

# Plot the most important feature map
plt.imshow(most_important_feature_map, cmap='viridis')
plt.colorbar()
plt.title('Most Important Feature Map')
plt.show()

# Optional: Normalize and plot the feature map as an image
norm_feature_map = (most_important_feature_map - np.min(most_important_feature_map)) / (np.max(most_important_feature_map) - np.min(most_important_feature_map))
plt.imshow(norm_feature_map, cmap='gray')
plt.title('Normalized Feature Map as Image')
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
import IPython.display as ipd

# Assuming most_important_feature_map is already computed
# Normalize the feature map
norm_feature_map = (most_important_feature_map - np.min(most_important_feature_map)) / (np.max(most_important_feature_map) - np.min(most_important_feature_map))

# Define parameters for the inverse STFT
hop_length = 256  # Number of samples between successive frames
win_length = 512  # Each frame of audio is windowed

# Convert the normalized feature map back to audio using the inverse STFT
# Transpose because librosa expects time on the first axis and frequency on the second axis
audio_data = librosa.istft(norm_feature_map.T, hop_length=hop_length, win_length=win_length)

# Define the sample rate
sample_rate = 22050  # 22.05 kHz, standard for audio

# Save as WAV file using soundfile
sf.write('../data/features/feature_map.wav', audio_data, sample_rate)

# Load and play the audio file using IPython.display.Audio
audio, sr = librosa.load('../data/features/feature_map.wav', sr=sample_rate)
ipd.Audio(audio, rate=sr)


# %%
import tensorflow as tf
import numpy as np

# Set a seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define the shape of the filters (3x3, 1 input channel, 64 filters)
filter_shape = (3, 3, 1, 64)

# Initialize custom weights using a predefined method (e.g., random values)
custom_filters = np.random.randn(*filter_shape).astype(np.float32)

# Initialize biases (optional, here set to zero)
custom_biases = np.zeros(64, dtype=np.float32)

# Define the Sequential model with a Conv2D layer
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(28, 28, 1))
])

# Build the model by providing a sample input
model.build(input_shape=(None, 28, 28, 1))

# Access the Conv2D layer
conv_layer = model.layers[0]

# Set the custom weights to the Conv2D layer
conv_layer.set_weights([custom_filters, custom_biases])

# Verify the weights have been set correctly
filters, biases = conv_layer.get_weights()

print(f"Filters shape: {filters.shape}")  # Should be (3, 3, 1, 64)
print(f"Biases shape: {biases.shape}")    # Should be (64,)

# Print the values of the custom filters
for i in range(filters.shape[-1]):
    print(f"Filter {i + 1} values:")
    print(filters[:, :, :, i])
    print()  # Print a newline for better readability


# %%
