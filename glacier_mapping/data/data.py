"""
Custom Dataset for Training
"""
#!/usr/bin/env python
import glob
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch

def fetch_loaders(processed_dir, batch_size=32):
    """
    DataLoaders for Training / Validation

    :param processed_dir: Directory with the processed data
    :param batch_size: The size of each batch during training. Defaults to 32.
    """
    train_dataset = GlacierDataset(processed_dir / "train")
    val_dataset = GlacierDataset(processed_dir / "dev")

    return {
        "train": DataLoader(train_dataset, batch_size=batch_size, num_workers=8),
        "val": DataLoader(val_dataset, batch_size=batch_size, num_workers=3)
    }


class GlacierDataset(Dataset):
    """
    Custom Dataset for Glacier Data

    Indexing the i^th element returns the underlying image and the associated
    binary mask
    """

    def __init__(self, folder_path):
        """Initialize dataset."""

        self.img_files = glob.glob(os.path.join(folder_path, '*img*'))
        self.mask_files = [s.replace("img", "mask") for s in self.img_files]

    def __getitem__(self, index):
        img_path = self.img_files[index]
        mask_path = self.mask_files[index]
        data = np.load(img_path)
        label = np.load(mask_path)

        return torch.from_numpy(data).float(), torch.from_numpy(label).float()

    def __len__(self):
        return len(self.img_files)
