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
audio_dir = (
    "../data/SoundMeters_Ingles_Primary"
)

# %%
features_dir = "../data/features"
os.makedirs(features_dir, exist_ok=True)

# %%
clusters_dir = "../data/clusters"
os.makedirs(clusters_dir, exist_ok=True)

# %%
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

# %%
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
