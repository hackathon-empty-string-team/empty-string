from torch.utils.data import DataLoader
from AudioDataset import OverlappingAudioDataset, NonOverlappingAudiDataset


# Initialize dataset
# dataset = OverlappingAudioDataset(root_dir='/python/data/SoundMeters_Ingles_Primary-20240519T132658Z-009/SoundMeters_Ingles_Primary')
dataset = NonOverlappingAudioDataset(root_dir='/python/data/SoundMeters_Ingles_Primary-20240519T132658Z-009/SoundMeters_Ingles_Primary')

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Function to test and print dataset shapes
def test_dataloader(dataloader):
    for i, batch in enumerate(dataloader):
        print(f"Batch {i+1} shape: {batch.shape}")
        if i == 2:  # Print the first three batches
            break

# Run the test
test_dataloader(dataloader)
