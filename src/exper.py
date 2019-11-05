#!/usr/bin/env python
import argparse
import os
import pathlib
import torch
import wandb

from src.dataset import GlacierDataset, loader
from src.trainer import Trainer
from src.unet import Unet
from src.utils import  get_opts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "-m",
            "--message",
            type=str,
            default="",
            help="Add a message to the experiment",
    )
    parser.add_argument(
            "-c",
            "--conf_name",
            type=str,
            default="defaults",
            help="name of conf file in config/",
    )
    parser.add_argument(
            "-o",
            "--output_dir",
            type=str,
            help="where the runs data should be stored"
    )

    # setup directories for output
    parsed_opts = parser.parse_args()
    output_path = pathlib.Path(parsed_opts.output_dir).resolve()
    if not output_path.exists():
        output_path.mkdir()

    opts = get_opts(parsed_opts.conf_name)
    opts["train"]["output_path"] = output_path
    os.environ["WANDB_MODE"] = "dryrun"
    wandb.init(dir=str(output_path))
    wandb.config.update(opts.to_dict())
    wandb.config.update({"__message": parsed_opts.message})

    model = Unet(**opts["model"])
    train_loader = loader(opts["data"], opts["train"], mode="train")
    dev_loader = loader(opts["data"], opts["train"], mode="dev")

	# only dev as a start
    test_loader = None
    trainer = Trainer(
            model,
            opts["train"],
            train_loader,
            dev_loader,
            test_loader
    )

    trainer.train()
