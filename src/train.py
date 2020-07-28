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
from src.data.data import fetch_loaders
from src.utils.frame import Framework
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

def log_epoch(epoch, n_epochs, i, n, loss, batch_size):
    print(
        f"Epoch {epoch}/{n_epochs}, Training batch {i+1} of {int(len(train_dataset) // batch_size)}, Loss= {loss/batch_size:.5f}",
        end="\r",
        flush=True,
    )

def log_metrics(writer, metrics, avg_loss, epoch, stage="train"):
    print(f"\nT_Loss: {avg_loss:.5f}", end=" ")
    for k, v in metrics.items():
        print(f", {k}: v:.3f}", end=" ")
        writer.add_scalar(f"{stage}/Loss", avg_loss, epoch)
    for k, v in metrics.items():
        writer.add_scalar(f"{stage}/{str(k)}", v, epoch)



if __name__ == "__main__":
    args = get_args()
    data_dir = Path(os.environ["DATA_DIR"])
    conf = Dict(yaml.safe_load(open(args.conf, "r")))
    processed_dir = data_dir / "processed"

    loaders = fetch_loaders(processed_dir, batch_size)
    frame = Framework(
        model_opts=conf.model_opts,
        optimizer_opts=conf.optim_opts,
        metrics_opts=conf.metrics_opts,
        reg_opts=conf.reg_opts
    )

    # Tensorboard path
    writer = SummaryWriter(f"{data_dir}/runs/{args.run_name}/logs/")
    writer.add_text("Arguments", json.dumps(vars(args)))
    writer.add_text("Configuration Parameters", json.dumps(conf))

    for epoch in range(args.epochs):
        loss, metrics = 0, []

        for i, (x, y) in enumerate(loader["train"]):
            y_hat, _loss = frame.optimize(x, y)
            log_epoch(epoch, args.epochs, i, len(loader["train"].dataset), _loss, args.batch_size)
            loss += _loss
            metrics.append(frame.calculate_metrics(y_hat, y))

        # log training stuff
        log_metrics(writer, metrics, loss / len(train_dataset), epoch)

        # log validation stuff

        # save model


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
                metrics = frame.calculate_metrics(y_hat, y)
            else:
                metrics += frame.calculate_metrics(y_hat, y)
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
            out_dir=f"{data_dir}/runs/{args.run_name}/models/"
            frame.save(out_dir, epoch)

    writer.close()
