import torch.nn as nn

class VGGishAutoencoder(nn.Module):
    def __init__(self, input_channels=1, encoded_dim=128):
        super(VGGishAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),  # 4 times larger than 8
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # [batch_size, 32, 32, 48]
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 4 times larger than 16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # [batch_size, 64, 16, 24]
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 4 times larger than 32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # [batch_size, 128, 8, 12]
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 4 times larger than 64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # [batch_size, 256, 4, 6]
        )
        
        self.flatten_size = 256 * 4 * 6  # Calculate based on encoder output
        
        # Fully connected layers for the bottleneck
        self.fc1 = nn.Linear(self.flatten_size, encoded_dim)
        self.fc2 = nn.Linear(encoded_dim, self.flatten_size)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # 4 times larger than 64
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 4 times larger than 32
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 4 times larger than 16
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Use Sigmoid for the final layer to normalize the output
        )
    
    def forward(self, x):
        batch_size, num_examples, input_channels, height, width = x.shape
        
        x = x.view(batch_size * num_examples, input_channels, height, width)  # Merge batch and example dimensions
        
        x = self.encoder(x)
        
        x = x.view(batch_size * num_examples, -1)  # Flatten
        
        encoded = self.fc1(x)
        
        x = self.fc2(encoded)
        
        x = x.view(batch_size * num_examples, 256, 4, 6)  # Reshape to match the dimensions before decoding
        
        x = self.decoder(x)
        
        x = x.view(batch_size, num_examples, input_channels, 64, 96)  # Reshape back to the original input shape
        
        return encoded, x

