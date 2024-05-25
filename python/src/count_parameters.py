import torch
import torch.nn as nn
from SmallVGGish import SmallerVGGishAutoencoder
from VGGish import VGGishAutoencoder  # Ensure this is correctly imported from where you defined it

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Instantiate the models
smaller_vggish_model = SmallerVGGishAutoencoder(input_channels=1, encoded_dim=128)
vggish_model = VGGishAutoencoder(input_channels=1, encoded_dim=128)

# Calculate and print the number of trainable parameters
smaller_vggish_params = count_parameters(smaller_vggish_model)
vggish_params = count_parameters(vggish_model)

print(f"Number of trainable parameters in SmallerVGGishAutoencoder: {smaller_vggish_params}")
print(f"Number of trainable parameters in VGGishAutoencoder: {vggish_params}")
