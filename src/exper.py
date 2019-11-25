import logging
import os
import pickle
from argparse import ArgumentParser

import torch
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as T

from torch.utils.data import DataLoader

import preprocess
from trainer import Config, Trainer
from dataset import GlacierDataset
from unet import Unet


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--toy_data", default=True, type=bool,
                        help="whether to use the toy data or all the data")
    args = parser.parse_args()
    if args.toy_data:
        base_dir = '../data/toy_data'
    else:
        base_dir = '../data'

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    mask_used = 'debris_glaciers'
    normalize = True
    if mask_used == 'multi_class_glaciers':
      classes, multi_class = 3, True
    else:
      classes, multi_class = 1, False

    sat_channels_to_include = list(range(10))
    channels, depth = len(sat_channels_to_include), 4

    model = Unet(channels, classes, depth)
    model.to(device)

    config = Config(lr=0.0001, epochs=1, save_dir='../models/', save_freq=1, multi_class=multi_class)

    data_file = 'sat_data.csv'
    borders = False
    batch_size = 2

    # get normalization values
    img_transform = None
    if normalize:
      norm_data_file = os.path.join(base_dir, "normalization_data.pkl")
      norm_data = pickle.load(open(norm_data_file, "rb"))
      img_transform = T.Normalize(norm_data['mean'], norm_data['std'])

    train_dataset = GlacierDataset(base_dir, data_file, borders=borders,
                                 img_transform=img_transform,
                                 channels_to_inc=sat_channels_to_include, mask_used=mask_used)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                               num_workers=1,
                                               sampler=SubsetRandomSampler(range(5)))


    dev_dataset = GlacierDataset(base_dir, data_file, mode='dev', borders=borders,
                                 img_transform=img_transform,
                                 channels_to_inc=sat_channels_to_include, mask_used=mask_used)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size,
                                             shuffle=False, num_workers=1)

    test_dataset = GlacierDataset(base_dir, data_file, mode='test', borders=borders,
                                  img_transform=img_transform,
                                  channels_to_inc=sat_channels_to_include, mask_used=mask_used)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                              shuffle=False, num_workers=1)

    trainer = Trainer(model, config,
                      train_loader, dev_loader, test_loader,
                      device)

    trainer.train()
