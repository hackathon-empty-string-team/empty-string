import unittest
import os
import torch
from torch.utils.data import DataLoader
from AudioDataset import NonOverlappingAudioDataset
from custom_collate import custom_collate_fn

class TestNonOverlappingAudioDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Path to the directory containing test audio files
        cls.test_dir = '/python/data'  # Update this to your test audio directory

    def test_dataset_length(self):
        dataset = NonOverlappingAudioDataset(self.test_dir)
        self.assertGreater(len(dataset), 0, "Dataset should not be empty.")

    def test_getitem(self):
        dataset = NonOverlappingAudioDataset(self.test_dir)
        for i in range(len(dataset)):
            item = dataset[i]
            if item is not None:
                self.assertEqual(item.dim(), 4, "Each item should have 4 dimensions.")
                self.assertEqual(item.size(1), 1, "Channel dimension should be 1.")
                self.assertEqual(item.size(2), dataset.n_mels, f"Mel dimension should be {dataset.n_mels}.")
                self.assertEqual(item.size(3), dataset.num_frames, f"Frame dimension should be {dataset.num_frames}.")

    def test_collate_function(self):
        dataset = NonOverlappingAudioDataset(self.test_dir)
        dataloader = DataLoader(dataset, batch_size=4, collate_fn=custom_collate_fn)
        for batch in dataloader:
            if batch.size(0) > 0:  # If batch is not empty
                self.assertEqual(batch.dim(), 5, "Batch should have 5 dimensions.")
                self.assertEqual(batch.size(2), 1, "Channel dimension should be 1.")
                self.assertEqual(batch.size(3), dataset.n_mels, f"Mel dimension should be {dataset.n_mels}.")
                self.assertEqual(batch.size(4), dataset.num_frames, f"Frame dimension should be {dataset.num_frames}.")

if __name__ == '__main__':
    unittest.main()
