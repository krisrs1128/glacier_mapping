#!/usr/bin/env python
"""
Training/Validation Pipeline:
    1- Initialize loaders (train & validation)
        1.1-Pass all params onto both loaders
    2- Initialize the framework
    3- Train Loop e epochs
        3.1 Pass entire data loader through epoch
        3.2 Iterate over dataloader with specific batch
    4- Log Epoch level train loss, test loss, metrices, image prediction each s step.
    5- Save checkpoints after 5 epochs
    6- -n is the required parameter (name_of_the_run)
    6- models are saved in path/models/name_of_the_run
    7- tensorboard is saved in path/runs/name_of_the_run
"""
from addict import Dict
from pathlib import Path
from src.data.data import GlacierDataset
from src.utils.frame import Framework
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import json
import math
import numpy as np
import os
import torch
import torchvision
import yaml

np.random.seed(7)


def get_num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def get_args():

    parser = argparse.ArgumentParser(
        description="Train the UNet on images and target masks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-n", "--name", type=str, help="Name of run", dest="run_name", required=True
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=500,
        help="Number of epochs (Default 500)",
        dest="epochs",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=16,
        help="Batch size (Default 16)",
        dest="batch_size",
    )
    parser.add_argument(
        "-s",
        "--save_every",
        type=int,
        default=5,
        help="Save every n epoch (Default 5)",
        dest="save_every",
    )

    conf_dir = Path(os.environ["ROOT_DIR"], "conf")
    parser.add_argument(
        "-c",
        "--conf",
        type=str,
        default=str(conf_dir / "train.yaml"),
        help="Configuration File for training",
        dest="conf",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    data_dir = Path(os.environ["DATA_DIR"])
    conf = Dict(yaml.safe_load(open(args.conf, "r")))
    processed_dir = data_dir / "processed"

    train_dataset = GlacierDataset(processed_dir / "train")
    val_dataset = GlacierDataset(processed_dir / "dev")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=3
    )
    frame = Framework(
        model_opts=conf.model_opts,
        optimizer_opts=conf.optim_opts,
        metrics_opts=conf.metrics_opts,
        reg_opts=conf.reg_opts,
        out_dir=f"{data_dir}/runs/{args.run_name}/models/",
    )

    # Tensorboard path
    writer = SummaryWriter(f"{data_dir}/runs/{args.run_name}/logs/")
    writer.add_text("Arguments", json.dumps(vars(args)))
    writer.add_text("Configuration Parameters", json.dumps(conf))

    # Prepare image grid train/val, x,y to write in tensorboard
    _sample_train_images, _sample_train_labels = iter(train_loader).next()
    _sample_val_images, _sample_val_labels = iter(val_loader).next()

    # Write image to tensorboard
    _view_x_train = _sample_train_images[:, :, :, [2, 1, 0]]
    _view_x_train = _view_x_train.permute(0, 3, 1, 2)
    _view_x_train = (_view_x_train - _view_x_train.min()) / (
        _view_x_train.max() - _view_x_train.min()
    )
    train_img_grid = torchvision.utils.make_grid(_view_x_train, nrow=4)
    _labels = _sample_train_labels.permute(0, 3, 1, 2)
    train_label_grid = torchvision.utils.make_grid(_labels, nrow=4)
    writer.add_image("Train/image", train_img_grid)
    writer.add_image("Train/labels", train_label_grid)

    _view_x_val = _sample_val_images[:, :, :, [2, 1, 0]]
    _view_x_val = _view_x_val.permute(0, 3, 1, 2)
    _view_x_val = (_view_x_val - _view_x_val.min()) / (
        _view_x_val.max() - _view_x_val.min()
    )
    val_img_grid = torchvision.utils.make_grid(_view_x_val, nrow=4)
    _labels = _sample_val_labels.permute(0, 3, 1, 2)
    val_label_grid = torchvision.utils.make_grid(_labels, nrow=4)
    writer.add_image("Validation/image", val_img_grid)
    writer.add_image("Validation/labels", val_label_grid)

      # Saving json model file for the tool
    tool_dir = f"{data_dir}/runs/{args.run_name}/tool_files/"
    num_params = get_num_params(frame.model)

    tool_dict = {
        str(args.run_name): {
            "metadata": {"displayName": str(args.run_name)},
            "model": {
                "type": "pytorch",
                "numParameters": num_params,
                "name": conf.model_opts.name,
                "args": conf.model_opts.args,
                "inputShape": (512, 512, int(conf.model_opts.args.inchannels)),
                "fn": str(
                    Path(
                        f"{data_dir}/runs/{args.run_name}/models/", f"model_{args.epochs}.pt"
                    )
                ),
                "fineTuneLayer": 0,
                "process": "conf/postprocess.yaml",
            },
        }
    }

    tool_json = json.dumps(tool_dict, indent=4)

    with open(tool_dir + "model.json", "w") as model_tool:
        model_tool.write(tool_json)

    # Calculate number of batches once
    n_batches = int(math.ceil(len(train_dataset) / args.batch_size))

    for epoch in range(1, args.epochs + 1):
        ## Training loop
        loss = 0
        for i, (x, y) in enumerate(train_loader):
            frame.set_input(x, y)
            _loss = frame.optimize()
            print(
                f"Epoch {epoch}/{args.epochs}, Training batch {i+1} of {n_batches}, Loss= {_loss/args.batch_size:.5f}",
                end="\r",
                flush=True,
            )
            loss += _loss
            if i == 0:
                metrics = frame.calculate_metrics()
            else:
                metrics += frame.calculate_metrics()
        # Print and write scalars to tensorboard
        epoch_train_loss = loss / len(train_dataset)
        print(f"\nT_Loss: {epoch_train_loss:.5f}", end=" ")
        for i, k in enumerate(conf.metrics_opts):
            print(f", {k}: {metrics[i]/len(train_dataset):.3f}", end=" ")
        writer.add_scalar("Loss/train", epoch_train_loss, epoch)
        for i, k in enumerate(conf.metrics_opts):
            writer.add_scalar("Train/" + str(k), metrics[i] / len(train_dataset), epoch)
        # Write Images to tensorboard
        if epoch % args.save_every == 0:
            y_hat = frame.infer(_sample_train_images.to(frame.device))
            y_hat = torch.sigmoid(y_hat)
            _preds = y_hat.permute(0, 3, 2, 1)
            pred_grid = torchvision.utils.make_grid(_preds, nrow=4)
            writer.add_image("Train/predictions", pred_grid, epoch)

        ## Validation loop
        loss = 0
        for i, (x, y) in enumerate(val_loader):
            y_hat = frame.infer(x.to(frame.device))
            _loss = frame.calc_loss(y_hat.to(frame.device), y.to(frame.device)).item()
            loss += _loss
            if i == 0:
                metrics = frame.calculate_metrics()
            else:
                metrics += frame.calculate_metrics()
        epoch_val_loss = loss / len(val_dataset)
        frame.val_operations(epoch_val_loss)
        # Print and write scalars to tensorboard
        print(f"\nV_Loss: {epoch_val_loss:.5f}", end=" ")
        for i, k in enumerate(conf.metrics_opts):
            print(f", {k}: {metrics[i]/len(val_dataset):.3f}", end=" ")
        writer.add_scalar("Loss/val", epoch_val_loss, epoch)
        for i, k in enumerate(conf.metrics_opts):
            writer.add_scalar(
                "Validation/" + str(k), metrics[i] / len(val_dataset), epoch
            )
        # Write images to tensorboard
        if epoch % args.save_every == 0:
            y_hat = frame.infer(_sample_val_images.to(frame.device))
            y_hat = torch.sigmoid(y_hat)
            _preds = y_hat.permute(0, 3, 2, 1)
            val_pred_grid = torchvision.utils.make_grid(_preds, nrow=4)
            writer.add_image("Validation/predictions", val_pred_grid, epoch)
        # Write combined loss graph to tensorboard
        writer.add_scalars(
            "Loss", {"train": epoch_train_loss, "val": epoch_val_loss}, epoch
        )
        print("\n")
        # Save model
        if epoch % args.save_every == 0:
            frame.save(frame.out_dir, epoch)

    writer.close()
