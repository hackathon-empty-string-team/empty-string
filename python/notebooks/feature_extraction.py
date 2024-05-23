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
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
import pickle


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
    output_tensor = conv_layer(input_tensor)
    
    # Apply global average pooling to get the output as a vector with 128 features
    output_vector = tf.keras.layers.GlobalAveragePooling2D()(output_tensor)
    
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



# %%
# Parameters
window_size = 1  # window size in seconds

# Path to the audio file
# currently this only works for one file (this can easily be scaled)
audio_file_path = '/python/data/SoundMeters_Ingles_Primary-20240519T132658Z-017/SoundMeters_Ingles_Primary/SM4XPRIZE_20240409_235202.wav'

# Load audio file
y, sr = librosa.load(audio_file_path, sr=None)

# Convert window size to samples
window_samples = int(window_size * sr)

# Create a 1-second chunk
y_window = y[:window_samples]

# Compute STFT
S = librosa.stft(y_window)
S_db = librosa.amplitude_to_db(np.abs(S))

# Define Gaussian parameters
x_means = np.linspace(0, S_db.shape[1], 5)
y_means = np.linspace(0, S_db.shape[0], 5)
std_dev_x = S_db.shape[1] / 10
std_dev_y = S_db.shape[0] / 10

test_array = np.ones((1025, 188))
# # Apply Gaussian masks to the spectrogram
# gaussian_spectrograms = apply_gaussian_to_spectrogram(S_db, x_means, y_means, std_dev_x, std_dev_y)

# Initialize dictionary to store results
results_dict = {
    'audio_file_path': audio_file_path,
    'window_size': window_size,
    'sampling_rate': sr,
    'spectrograms': {}
}


# %%
# Process each second of the audio file
for i in range(0, len(y), window_samples):
    y_window = y[i:i + window_samples]
    
    # Check if the window has the required length
    if len(y_window) < window_samples:
        break
    
    # Compute STFT
    S = librosa.stft(y_window)
    S_db = librosa.amplitude_to_db(np.abs(S))
    
    # Define Gaussian parameters
    x_means = np.linspace(0, S_db.shape[1], 5)
    y_means = np.linspace(0, S_db.shape[0], 5)
    std_dev_x = S_db.shape[1] / 10
    std_dev_y = S_db.shape[0] / 10
    
    # Apply Gaussian masks to the spectrogram
    gaussian_spectrograms = apply_gaussian_to_spectrogram(S_db, x_means, y_means, std_dev_x, std_dev_y)
    
    # Store results in the dictionary
    for key, data in gaussian_spectrograms.items():
        S_db_gaussian = data['S_db_gaussian']
        output_vector_np = apply_conv2d(S_db_gaussian)
        if key not in results_dict['spectrograms']:
            results_dict['spectrograms'][key] = []
        results_dict['spectrograms'][key].append({
            'S_db_gaussian': S_db_gaussian,
            'x_mean': data['x_mean'],
            'y_mean': data['y_mean'],
            'std_dev_x': data['std_dev_x'],
            'std_dev_y': data['std_dev_y'],
            'conv_output': output_vector_np
        })

# %%
# Example usage:
plot_random_spectrograms(results_dict)

# %%
# Save the results dictionary

# Directory to save features
features_dir = "../data/result_feature_extraction"
os.makedirs(features_dir, exist_ok=True)
with open('../data/result_feature_extraction/results_dict.pkl', 'wb') as f:
    pickle.dump(results_dict, f)

# %%
