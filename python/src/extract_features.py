from SmallVGGish import SmallerVGGishAutoencoder
from VGGish import VGGishAutoencoder
import torch
import numpy as np
import librosa

def load_model(model_path, device, input_channels=1, encoded_dim=128):
    model = VGGishAutoencoder(input_channels=input_channels, encoded_dim=encoded_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model('/python/data/vggish_autoencoder.pth', device)

def extract_features(file_path, model, device, target_height=64, target_width=96, overlapping=False):
    y, sr = librosa.load(file_path, sr=None)
    y = librosa.resample(y, orig_sr=sr, target_sr=16000)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=16000, n_mels=target_height)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    if overlapping:
        hop_length = int(16000 * 0.01)  # 10ms hop length for overlapping frames
    else:
        hop_length = target_width  # Frame length for non-overlapping frames

    # Frame the log mel spectrogram
    log_mel_examples = librosa.util.frame(log_mel_spectrogram, frame_length=target_width, hop_length=hop_length)
    log_mel_examples = log_mel_examples.transpose(2, 0, 1)  # Shape: (num_examples, height, width)
    log_mel_examples = torch.tensor(log_mel_examples).float().unsqueeze(1)  # Shape: (num_examples, 1, height, width)

    # Add batch dimension
    log_mel_examples = log_mel_examples.unsqueeze(0)  # Shape: (1, num_examples, 1, height, width)

    model.eval()
    with torch.no_grad():
        encoded, _ = model(log_mel_examples.to(device))
    return encoded.squeeze().cpu().numpy()

# Example usage
file_path = '/python/data/SoundMeters_Ingles_Primary-20240519T132658Z-009/SoundMeters_Ingles_Primary/SM4XPRIZE_20240409_193102.wav'
encoded_features = extract_features(file_path, model, device, overlapping=True)
print(f"Encoded features shape: {encoded_features.shape}")
