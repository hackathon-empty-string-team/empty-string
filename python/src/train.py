import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from SmallVGGish import SmallerVGGishAutoencoder
from VGGish import VGGishAutoencoder
from AudioDataset import OverlappingAudioDataset, NonOverlappingAudioDataset

def train_model(model, dataloader, num_epochs=1, learning_rate=1e-3):
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss().to(device)  # Move criterion to device

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0  # Track loss for the epoch
        for i, inputs in enumerate(dataloader):
            inputs = inputs.to(device)
            optimizer.zero_grad()
            encoded, decoded = model(inputs)
            
            loss = criterion(decoded, inputs)  # Compare the reconstructed output with the input
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    print("Finished Training")

model = VGGishAutoencoder(input_channels=1, encoded_dim=128)
# Example usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# select which type of audio dataset you want, either overlapping or not
dataset = OverlappingAudioDataset(root_dir='/python/data/SoundMeters_Ingles_Primary-20240519T132658Z-009/SoundMeters_Ingles_Primary')
# dataset = OverlappingAudioDataset(root_dir='/python/data') # run this only if you have enough compute to load the whole data directory
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)


train_model(model, dataloader, num_epochs=5)

# Save the trained model
torch.save(model.state_dict(), '/python/data/vggish_autoencoder.pth')
