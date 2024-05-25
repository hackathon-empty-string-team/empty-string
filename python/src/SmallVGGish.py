import torch.nn as nn

class SmallerVGGishAutoencoder(nn.Module):
    """
    A PyTorch implementation of a smaller VGGish-based Autoencoder for audio feature extraction and reconstruction.
    
    Attributes:
        encoder (nn.Sequential): The encoder part of the autoencoder, consisting of convolutional layers.
        flatten_size (int): The size of the flattened encoder output.
        fc1 (nn.Linear): The fully connected layer that compresses the encoded features.
        fc2 (nn.Linear): The fully connected layer that decompresses the encoded features.
        decoder (nn.Sequential): The decoder part of the autoencoder, consisting of transposed convolutional layers.
    
    Methods:
        forward(x): Forward pass through the autoencoder.
    """
    def __init__(self, input_channels=1, encoded_dim=128):
        """
        Initializes the SmallerVGGishAutoencoder with the given parameters.

        Args:
            input_channels (int): Number of input channels. Default is 1.
            encoded_dim (int): Dimension of the encoded feature vector. Default is 128.
        """
        super(SmallerVGGishAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # [batch_size, 8, 32, 48]
            
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # [batch_size, 16, 16, 24]
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # [batch_size, 32, 8, 12]
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)   # [batch_size, 64, 4, 6]
        )
        
        self.flatten_size = 64 * 4 * 6  # Calculate based on encoder output
        
        # Fully connected layers for the bottleneck
        self.fc1 = nn.Linear(self.flatten_size, encoded_dim)
        self.fc2 = nn.Linear(encoded_dim, self.flatten_size)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Use Sigmoid for the final layer to normalize the output
        )
    
    def forward(self, x):
        """
        Forward pass through the autoencoder.

        Args:
            x (torch.Tensor): Input tensor with shape [batch_size, num_examples, channels, height, width].
        
        Returns:
            tuple: Encoded features and reconstructed input tensor.
        """
        batch_size, num_examples, channels, height, width = x.shape
        # print(f"Input shape: {x.shape}")
        
        x = x.view(batch_size * num_examples, channels, height, width)  # Merge batch and example dimensions
        # print(f"After view (merge batch and example dims): {x.shape}")
        
        x = self.encoder(x)
        # print(f"After encoder: {x.shape}")
        
        x = x.view(batch_size * num_examples, -1)  # Flatten
        # print(f"After flatten: {x.shape}")
        
        encoded = self.fc1(x)
        # print(f"Encoded: {encoded.shape}")
        
        x = self.fc2(encoded)
        # print(f"After fc2: {x.shape}")
        
        x = x.view(batch_size * num_examples, 64, 4, 6)  # Reshape to match the dimensions before decoding
        # print(f"After reshape to match decoder input: {x.shape}")
        
        x = self.decoder(x)
        # print(f"After decoder: {x.shape}")
        
        x = x.view(batch_size, num_examples, 1, 64, 96)  # Reshape back to the original input shape
        # print(f"Final output shape: {x.shape}")
        
        return encoded, x

