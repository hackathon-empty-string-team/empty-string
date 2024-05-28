import numpy as np
import torch
import librosa
from torch.utils.data import Dataset

class SmallAudioDataset(Dataset):
    def __init__(self, file_paths, target_sr=16000, n_mels=64, num_frames=96):
        self.file_paths = file_paths
        self.target_sr = target_sr
        self.n_mels = n_mels
        self.num_frames = num_frames

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        y, sr = librosa.load(file_path, sr=None)
        y = librosa.resample(y, orig_sr=sr, target_sr=self.target_sr)
        
        # Compute log mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=self.target_sr, n_mels=self.n_mels)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        # Frame into non-overlapping examples
        hop_length = int(self.target_sr * 0.01)  # 10ms hop length
        window_length = int(self.target_sr * 0.025)  # 25ms window length
        log_mel_examples = librosa.util.frame(log_mel_spectrogram, frame_length=self.num_frames, hop_length=hop_length)
        
        # Reshape to match input dimensions for the model
        log_mel_examples = log_mel_examples.transpose(2, 0, 1)  # shape (num_examples, n_mels, num_frames)
        log_mel_examples = torch.tensor(log_mel_examples).float().unsqueeze(1)  # add channel dimension
        return log_mel_examples

