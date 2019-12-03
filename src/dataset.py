#!/usr/bin/env python
import numpy as np
from pathlib import Path
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as T

import src.utils as utils
torch.manual_seed(10)

class GlacierDataset(Dataset):
    def __init__(self, base_dir, data_file, channels_to_inc=None, img_transform=None,
                 mode='train', borders=False, use_cropped=True, use_snow_i=False,
                 mask_used='glacier'):
        super().__init__()
        self.base_dir = base_dir
        data_path = Path(base_dir, data_file)
        self.data = pd.read_csv(data_path)
        self.data = self.data[self.data.train == mode]
        if mask_used == 'debris_glaciers':
            self.data = self.data[self.data.pseudo_debris_perc > 0]
        self.img_transform = img_transform
        self.borders = borders
        self.use_cropped = use_cropped
        self.use_snow_i = use_snow_i
        self.channels_to_inc = channels_to_inc
        self.mode = mode
        self.mask_used = mask_used

    def __getitem__(self, i):
        pathes = ['img_path', 'mask_path', 'border_path']
        image_path, mask_path, border_path = self.data.iloc[i][pathes]

        image_path = Path(self.base_dir, image_path)
        mask_path = Path(self.base_dir, mask_path)

        if self.use_cropped:
            cropped_pathes = ['cropped_path', 'cropped_label']
            cropped_img_path, cropped_label_path = self.data.iloc[i][cropped_pathes]
            cropped_img_path = Path(self.base_dir, cropped_img_path)
            cropped_label_path = Path(self.base_dir, cropped_label_path)

            cropped_img = np.load(cropped_img_path)
            mask_path = cropped_label_path if (
                self.mode == 'train') else mask_path
            image_path = cropped_img_path

        img = np.load(image_path)
        img = T.ToTensor()(img)

        if self.channels_to_inc is not None:
            img = img[self.channels_to_inc]

        if (self.borders) and (not pd.isnull(border_path)):
            border_path = Path(self.base_dir, border_path)
            border = np.load(border_path)
            border = np.expand_dims(border, axis=0)
            img = np.concatenate((img, border), axis=0)
            img = torch.from_numpy(img)

        if self.use_snow_i:
            snow_index = utils.get_snow_index(img)
            snow_index = np.expand_dims(snow_index, axis=0)
            img = np.concatenate((img, snow_index), axis=0)
            img = torch.from_numpy(img)

        if self.img_transform is not None:
            img = self.img_transform(img)

        # default is 'glaciers' for original labels 
        mask = np.load(mask_path)
        if self.mask_used == 'debris_glaciers':
            mask = utils.get_debris_glaciers(img, mask)
            return img, mask.astype(np.float32)
        elif self.mask_used == 'multi_class_glaciers':
            mask = utils.merge_mask_snow_i(img, mask.astype(np.int64))
        return img, mask.astype(np.float32)

    def __len__(self):
        return len(self.data)



def loader(data_opts, train_opts, img_transform, mode="train"):
  """
  Loader for Experiment
  """
  dataset = GlacierDataset(
    data_opts["path"],
    data_opts["metadata"],
    use_snow_i=data_opts["use_snow_i"],
    channels_to_inc=data_opts["channels_to_inc"],
    mask_used=data_opts["mask_used"],
    img_transform=img_transform,
    mode=mode,
    borders=data_opts["borders"]
  )

  if data_opts.load_limit == -1:
    sampler, shuffle = None, train_opts["shuffle"]
  else:
    sampler, shuffle = SubsetRandomSampler(range(data_opts.load_limit)), False

  return DataLoader(
    dataset,
    sampler=sampler,
    batch_size=train_opts["batch_size"],
    shuffle=shuffle,
    num_workers=train_opts["num_workers"],
    drop_last=True
  )
