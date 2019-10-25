#!/usr/bin/env python
from comet_ml import OfflineExperiment
import argparse
import torch

from src.trainer import Config, Trainer
from src.dataset import GlacierDataset
from src.unet import Unet


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

    opts = get_opts(parsed_opts.conf_name)
    exp = OfflineExperiment(offline_directory=str(output_path))
    self.exp.log_parameters(opts)

    model = Unet(
            opts["model"]["channels"],
            opts["model"]["classes"],
            opts["model"]["net_depth"]
    )

    train_loader = loader(opts["data"], opts["train"], mode="train")
    dev_loader = loader(opts["data"], opts["train"], mode="dev")

	# only dev as a start
    test_loader = None
    trainer = Trainer(
            exp,
            model,
            opts["train"],
			train_loader,
            dev_loader,
            test_loader
    )

    trainer.train()
