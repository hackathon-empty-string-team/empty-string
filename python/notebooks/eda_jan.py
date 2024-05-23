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
# ## New approach for data engineering

# %%


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



# %%
