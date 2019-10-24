#!/usr/bin/env python
import logging

import torch

from trainer import Config, Trainer
from dataset import GlacierDataset
from unet import Unet


if __name__ == '__main__':
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	channels, classes, depth = 11, 1, 4

	model = Unet(channels, classes, depth)
	model.to(device)

	config = Config(lr=0.0001, epochs=1, save_dir='./', save_freq=1)

	base_dir = '../data'
	data_file = 'sat_data.csv'
	borders = True
	batch_size = 4

	train_dataset = GlacierDataset(base_dir, data_file, mode='train', borders=borders)
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                          	  shuffle=True, num_workers=1)

	dev_dataset = GlacierDataset(base_dir, data_file, mode='dev', borders=borders)
	dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=batch_size,
                                          	  shuffle=True, num_workers=1)
	# only dev as a start
	test_loader = None

	trainer = Trainer(model, config,
					  train_loader, dev_loader, test_loader,
					  device)

	trainer.train()
