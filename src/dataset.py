import os
import math

import numpy as np
import pandas as pd

from torch.utils.data import Dataset
import torchvision.transforms as T

import utils


class GlacierDataset(Dataset):
    def __init__(self, base_dir, data_file, channels_to_inc=None, img_transform=None, mode='train',
                 borders=False, use_cropped=True, use_snow_i=True):
        super().__init__()
        self.base_dir = base_dir
        data_path = os.path.join(base_dir, data_file)
        self.data = pd.read_csv(data_path)
        self.data = self.data[self.data.train == mode]
        self.img_transform = img_transform
        self.borders = borders
        self.use_cropped = use_cropped
        self.use_snow_i = use_snow_i
        self.channels_to_inc = channels_to_inc
        self.mode = mode

    def __getitem__(self, i):
        pathes = ['img_path', 'mask_path', 'border_path']
        image_path, mask_path, border_path = self.data.iloc[i][pathes]

        image_path = os.path.join(self.base_dir, image_path)
        mask_path = os.path.join(self.base_dir, mask_path)

        if self.use_cropped:
            cropped_pathes = ['cropped_path', 'cropped_label']
            cropped_img_path, cropped_label_path = self.data.iloc[i][cropped_pathes]
            cropped_img_path = os.path.join(self.base_dir, cropped_img_path)
            cropped_label_path = os.path.join(
                self.base_dir, cropped_label_path)

            cropped_img = np.load(cropped_img_path)
            mask_path = cropped_label_path if (
                self.mode == 'train') else mask_path
            image_path = cropped_img_path

        img = np.load(image_path)
        if self.img_transform is not None:
            img = self.img_transform(img)
        else:
            img = T.ToTensor()(img)

        if self.channels_to_inc is not None:
            img = img[self.channels_to_inc]

        if (self.borders) and (not pd.isnull(border_path)):
            border_path = os.path.join(self.base_dir, border_path)
            border = np.load(border_path)
            border = np.expand_dims(border, axis=0)
            img = np.concatenate((img, border), axis=0)

        if self.use_snow_i:
            snow_index = utils.get_snow_index(img, thresh=0.6)
            snow_index = np.expand_dims(snow_index, axis=0)
            img = np.concatenate((img, snow_index), axis=0)

        return img, np.load(mask_path).astype(np.float32)

    def __len__(self):
        return len(self.data)
