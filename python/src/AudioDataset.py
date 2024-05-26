import os
import glob
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

class OverlappingAudioDataset(Dataset):
    """
    A custom PyTorch Dataset class for loading audio files and converting them into overlapping log mel spectrograms.

    Attributes:
        root_dir (str): Root directory containing the audio files.
        target_sr (int): Target sample rate for audio files. Default is 16000.
        n_mels (int): Number of mel bands to generate. Default is 64.
        num_frames (int): Number of frames in each example. Default is 96.

    Methods:
        __len__(): Returns the number of audio files in the dataset.
        __getitem__(idx): Returns the log mel spectrogram examples for the audio file at the specified index.
    """
    def __init__(self, root_dir, target_sr=16000, n_mels=64, num_frames=96):
        """
        Initializes the OverlappingAudioDataset with the given parameters and loads the paths of audio files.

        Args:
            root_dir (str): Root directory containing the audio files.
            target_sr (int): Target sample rate for audio files. Default is 16000.
            n_mels (int): Number of mel bands to generate. Default is 64.
            num_frames (int): Number of frames in each example. Default is 96.
        """
        self.file_paths = glob.glob(os.path.join(root_dir, '**/*.wav'), recursive=True)
        self.target_sr = target_sr
        self.n_mels = n_mels
        self.num_frames = num_frames

    def __len__(self):
        """
        Returns the number of audio files in the dataset.

        Returns:
            int: Number of audio files.
        """
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        Returns the log mel spectrogram examples for the audio file at the specified index.

        Args:
            idx (int): Index of the audio file.

        Returns:
            torch.Tensor: Log mel spectrogram examples with shape (num_examples, 1, n_mels, num_frames).
        """
        file_path = self.file_paths[idx]
        y, sr = librosa.load(file_path, sr=None)
        y = librosa.resample(y, orig_sr=sr, target_sr=self.target_sr)
        
        # Compute log mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=self.target_sr, n_mels=self.n_mels)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        # Frame into overlapping examples
        hop_length = int(self.target_sr * 0.01)  # 10ms hop length
        log_mel_examples = librosa.util.frame(log_mel_spectrogram, frame_length=self.num_frames, hop_length=hop_length)
        
        # Reshape to match input dimensions for the model
        log_mel_examples = log_mel_examples.transpose(2, 0, 1)  # shape (num_examples, n_mels, num_frames)
        log_mel_examples = torch.tensor(log_mel_examples).float().unsqueeze(1)  # add channel dimension

        return log_mel_examples

