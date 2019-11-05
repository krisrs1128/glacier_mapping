import logging
from argparse import ArgumentParser
import logging

import torch
from torch.utils.data.sampler import SubsetRandomSampler

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
    sat_channels_to_include = [0, 1, 2, 3, 4]
    channels, classes, depth = len(sat_channels_to_include) + 1, 1, 4

    model = Unet(channels, classes, depth)
    model.to(device)

    config = Config(lr=0.0001, epochs=1, save_dir='../models/', save_freq=1)

    data_file = 'sat_data.csv'
    borders = False
    batch_size = 2

    train_dataset = GlacierDataset(base_dir, data_file, mode='train', borders=borders,
                                   channels_to_inc=sat_channels_to_include)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               num_workers=1,
                                               sampler=SubsetRandomSampler(range(5)))

    dev_dataset = GlacierDataset(base_dir, data_file, mode='dev', borders=borders,
                                 channels_to_inc=sat_channels_to_include)
    dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=batch_size,
                                             shuffle=False, num_workers=1)

    test_dataset = GlacierDataset(base_dir, data_file, mode='test', borders=borders,
                                  channels_to_inc=sat_channels_to_include)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                              shuffle=False, num_workers=1)

    trainer = Trainer(model, config,
                      train_loader, dev_loader, test_loader,
                      device)

    trainer.train()
