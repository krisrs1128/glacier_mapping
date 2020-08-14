#!/usr/bin/env python
"""
Training/Validation Pipeline:
    1- Initialize loaders (train & validation)
        1.1-Pass all params onto both loaders
    2- Initialize the framework
    3- Train Loop args.e epochs
        3.1 Pass entire data loader through epoch
        3.2 Iterate over dataloader with specific batch
    4- Log Epoch level train loss, test loss, metrices, image prediction each s step.
    5- Save checkpoints after 5 epochs
    6- -n is the required parameter (name_of_the_run)
    6- models are saved in path/models/name_of_the_run
    7- tensorboard is saved in path/runs/name_of_the_run
"""
from pathlib import Path
import argparse
import os
import numpy as np
import pandas as pd
from torchvision.utils import make_grid
import torch
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


def train_epoch(loader, frame, metrics_opts):
    """
    Train model for One Epoch

    This makes one pass through a dataloader and updates the model in the
    associated frame.

    :param loader (DataLoader): A pytorch DataLoader containing x,y pairs
      with which to train the model.
    :param frame (Framework): A Framework object wrapping both the model and the
      optimization setup.
    :param metrics_opts (dict): A dictionary whose keys specify which metrics to
      compute on the predictions from the model.
    :return: A tuple containing the average epoch loss and the metrics on the
      training set.
    """
    loss, metrics = 0, []
    for x, y in loader:
        y_hat, _loss = frame.optimize(x, y)
        loss += _loss

        y_hat = torch.sigmoid(y_hat)
        metrics_ = frame.metrics(y_hat, y, metrics_opts)
        metrics.append(metrics_)

    return loss / len(loader.dataset), metrics


def validate(loader, frame, metrics_opts):
    """
    Compute Metrics on a Validation Loader

    To honestly evaluate a model, we should compute its metrics on a validation
    dataset. This runs the model in frame over the data in loader, compute all
    the metrics specified in metrics_opts.

    :param loader (DataLoader): A pytorch DataLoader containing x,y pairs
      with which to validate the model.
    :param frame (Framework): A Framework object wrapping both the model and the
      optimization setup.
    :param metrics_opts (dict): A dictionary whose keys specify which metrics to
      compute on the predictions from the model.
    :return: A tuple containing the average validation loss and the metrics on the
      validation set.
    """
    loss, metrics = 0, []
    for x, y in loader:
        y_hat = frame.infer(x)
        loss += frame.calc_loss(y_hat, y).item()

        y_hat = torch.sigmoid(y_hat)
        metrics_ = frame.metrics(y_hat, y, metrics_opts)
        metrics.append(metrics_)

    return loss / len(loader.dataset), metrics


def log_batch(epoch, n_epochs, i, n, loss, batch_size):
    """
    Helper to log a training batch

    :param epoch: Current epoch
    """
    print(
        f"Epoch {epoch}/{n_epochs}, Training batch {i+1} of {int(n) // batch_size}, Loss = {loss/batch_size:.5f}",
        end="\r",
        flush=True,
    )


def log_metrics(writer, metrics, avg_loss, epoch, stage="train"):
    """ Log metrics for tensorboard

    A function that logs metrics from training and testing to tensorboard

    Args:
        writer(SummaryWriter): The tensorboard summary object
        metrics(Dict): Dictionary of metrics to record
        avg_loss(float): The average loss across all epochs
        epoch(int): Total number of training cycles
        stage(String): Train/Val

    """
    metrics = dict(pd.DataFrame(metrics).mean())
    writer.add_scalar(f"{stage}/Loss", avg_loss, epoch)
    for k, v in metrics.items():
        writer.add_scalar(f"{stage}/{str(k)}", v, epoch)


def log_images(writer, frame, batch, epoch, stage="train"):
    """ Log images for tensorboard

    Args:
        writer (Tensorboard writer): Class to write images
        frame: Image frame to log
        batch: Image batch to log
        epoch: Number of epochs
        stage: specified pipeline stage

    Return:
        Images Logged onto tensorboard
    """
    pm = lambda x: x.permute(0, 3, 2, 1)
    squash = lambda x: (x - x.min()) / (x.max() - x.min())

    x, y = batch
    y_hat = torch.sigmoid(frame.infer(x))
    if epoch == 0:
        writer.add_image(f"{stage}/x", make_grid(pm(squash(x))), epoch)
        writer.add_image(f"{stage}/y", make_grid(pm(y)), epoch)

    writer.add_image(f"{stage}/y_hat", make_grid(pm(y_hat)), epoch)
