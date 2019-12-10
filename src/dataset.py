#!/usr/bin/env python
import numpy as np
from pathlib import Path
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as T

import src.utils as utils
import src.augmentations as TA
torch.manual_seed(10)

class GlacierDataset(Dataset):
    def __init__(self, base_dir, data_file, channels_to_inc=None, img_transform=None,
                 mode='train', borders=False, use_cropped=True, use_snow_i=False,
                 use_elev=True, use_slope=True, mask_used='glacier', country='all', year='all'):
        super().__init__()
        self.base_dir = base_dir
        data_path = Path(base_dir, data_file)
        self.data = pd.read_csv(data_path)
        self.data = self.data[self.data.train == mode]
        if mask_used == 'debris_glaciers':
            self.data = self.data[self.data.pseudo_debris_perc > 0]
        if country != 'all':
            self.data = self.data[self.data["country"].isin(country)]
        if year != 'all':
            self.data = self.data[self.data["year"].isin(year)]
        self.img_transform = img_transform
        self.borders = borders
        self.use_cropped = use_cropped
        self.use_snow_i = use_snow_i

        if channels_to_inc is not None:
            self.channels_to_inc = channels_to_inc[:]
        else: self.channels_to_inc = list(range(10))
        if use_slope: self.channels_to_inc.append(11)
        if use_elev: self.channels_to_inc.append(10)
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
            mask_path = cropped_label_path
            image_path = cropped_img_path

        img = np.load(image_path)
        img = T.ToTensor()(img)

        # get snow index before filtering the data
        snow_index = utils.get_snow_index(img)
        img = img[self.channels_to_inc]

        if self.use_snow_i:
            snow_index = np.expand_dims(snow_index, axis=0)
            img = np.concatenate((img, snow_index), axis=0)
            img = torch.from_numpy(img)

        if (self.borders) and (not pd.isnull(border_path)):
            border_path = Path(self.base_dir, border_path)
            border = np.load(border_path)
            border = np.expand_dims(border, axis=0)
            img = np.concatenate((img, border), axis=0)
            img = torch.from_numpy(img)

        if self.img_transform is not None:
            img = self.img_transform(img)

        # default is 'glaciers' for original labels 
        mask = np.load(mask_path)
        if self.mask_used == 'debris_glaciers':
            mask = utils.get_debris_glaciers(img, mask)
        elif self.mask_used == 'multi_class_glaciers':
            mask = utils.merge_mask_snow_i(img, mask.astype(np.int64))
        return img, mask.astype(np.float32)

    def __len__(self):
        return len(self.data)

class AugmentedGlacierDataset(GlacierDataset):
    def __init__(self, *args, hflip=0.5, vflip=0.5, rot_p=0.5, rot=30, **kargs):

        super().__init__(*args, **kargs)
        self.hflip = hflip
        self.vflip = vflip
        self.rot_p = rot_p
        self.rot = (-rot, rot)

    def __getitem__(self, i):
        img, mask = super().__getitem__(i)
        return self.augment(img, mask)

    def augment(self, img, mask):
        img = TA.to_numpy_img(img)
        img, mask = TA.rotate(img, mask, self.rot, p=self.rot_p)
        img, mask = TA.flip(img, mask, 0, self.vflip)
        img, mask = TA.flip(img, mask, 1, self.hflip)
        img = np.moveaxis(img, -1, 0)

        return img, mask

def loader(data_opts, train_opts, augment_opts, img_transform, mode="train"):
  """
  Loader for Experiment
  """
  data_args = [data_opts["path"], data_opts["metadata"]]
  data_kargs = {"use_snow_i":data_opts["use_snow_i"],
                "channels_to_inc":data_opts["channels_to_inc"],
                "mask_used":data_opts["mask_used"],
                "img_transform":img_transform,
                "mode":mode,
                "borders":data_opts["borders"],
                "year":data_opts["year"],
                "country":data_opts["country"],
                "hflip":augment_opts["hflip"],
                "vflip":augment_opts["vflip"],
                "rot_p":augment_opts["rotate_prop"],
                "rot":augment_opts["rotate_degree"]}

  dataset = AugmentedGlacierDataset(*data_args, **data_kargs)

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
