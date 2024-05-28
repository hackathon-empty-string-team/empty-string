import os
import glob
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
from SmallVGGish import SmallerVGGishAutoencoder
from VGGish import VGGishAutoencoder
from SmallAudioDataset import SmallAudioDataset

def train_on_single_batch(model, dataloader, num_epochs=100, learning_rate=1e-4):
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(num_epochs):
        for i, inputs in enumerate(dataloader):
            inputs = inputs.to(device)
            optimizer.zero_grad()
            encoded, decoded = model(inputs)
            
            # Ensure shapes match before calculating loss
            # print(f"Input shape: {inputs.shape}")
            # print(f"Decoded shape: {decoded.shape}")
            
            loss = criterion(decoded, inputs)  # Compare the reconstructed output with the input
            loss.backward()
            optimizer.step()
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

            if epoch % 10 == 9:
                print(f"After {epoch + 1} epochs, Loss: {loss.item():.4f}")

# Get a single file for overfitting test
root_dir = '/python/data/SoundMeters_Ingles_Primary-20240519T132658Z-009/SoundMeters_Ingles_Primary'
file_paths = glob.glob(os.path.join(root_dir, '**/*.wav'), recursive=True)[:1]  # Use only one file

# Create a small dataset and dataloader
small_dataset = SmallAudioDataset(file_paths)
small_dataloader = DataLoader(small_dataset, batch_size=4, shuffle=True)

# Print a batch to verify
for inputs in small_dataloader:
    print(f"Sample input shape: {inputs.shape}")
    break

# Example usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VGGishAutoencoder(input_channels=1, encoded_dim=128)
model.to(device)

train_on_single_batch(model, small_dataloader, num_epochs=100)
