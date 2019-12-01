#!/usr/bin/env python
import argparse
import os
import pathlib
import torch
import wandb

from src.preprocess import get_normalization
from src.dataset import GlacierDataset, loader
from src.trainer import Trainer
from src.unet import Unet
from src.utils import  get_opts

def induce_config(data_config):
    inchannels = (len(data_config.channels_to_inc) +
                  data_config.borders +
                  data_config.use_snow_i)

    if data_config.mask_used == "multi_class_glaciers":
        outchannels, multiclass = 2, True
    else: outchannels, multiclass = 1, False

    return inchannels, outchannels, multiclass


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
    inchannels, outchannels, multiclass = induce_config(opts["data"]) 
    opts["model"]["inchannels"] = inchannels
    opts["model"]["outchannels"] = outchannels
    opts["train"]["multiclass"] = multiclass

                                      
    os.environ["WANDB_MODE"] = "dryrun"
    wandb.init(dir=str(output_path))
    wandb.config.update(opts.to_dict())
    wandb.config.update({"__message": parsed_opts.message})
    
    img_transform = get_normalization(opts["data"])
    model = Unet(**opts["model"])
    train_loader = loader(opts["data"], opts["train"], mode="train", img_transform=img_transform)
    dev_loader = loader(opts["data"], opts["train"], mode="dev", img_transform=img_transform)
    test_loader = loader(opts["data"], opts["train"], mode="test", img_transform=img_transform)

    trainer = Trainer(
            model,
            opts["train"],
            train_loader,
            dev_loader,
            test_loader
    )
    trainer.train()
