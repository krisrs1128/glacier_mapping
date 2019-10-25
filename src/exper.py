#!/usr/bin/env python
import logging
import argparse
import torch

from trainer import Config, Trainer
from dataset import GlacierDataset
from unet import Unet


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "-m",
            "--message",
            type=str,
            default="",
            help="Add a message to the commet experiment",
    )
    parser.add_argument(
        "-c",
        "--conf_name",
        type=str,
        default="defaults",
        help="name of conf file in config/ | may ommit the .yaml extension",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        help="where the run's data should be stored ; used to resume",
    )

    # setup directories for output
    parsed_opts = parser.parse_args()
    output_path = Path(parsed_opts.output_dir).resolve()
    if not output_path.exists():
        output_path.mkdir()
    exp = OfflineExperiment(offline_directory=str(output_path))
    opts = get_opts(parsed_opts.conf_name)

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
