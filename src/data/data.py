#!/usr/bin/env python
from torch.utils.data import Dataset
import glob
import numpy as np
import os
import torch


class GlacierDataset(Dataset):

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
    
