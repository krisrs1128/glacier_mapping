#!/usr/bin/env python
"""
Training/Validation Module

The overall training and validation pipeline has the following structure,

* Initialize loaders (train & validation)
* Initialize the framework
* Train Loop args.e epochs
* Log Epoch level train loss, test loss, metrices, image prediction each s step.
* Save checkpoints after save_every epochs
* models are saved in path/models/name_of_the_run
* tensorboard is saved in path/runs/name_of_the_run
"""
from pathlib import Path
import argparse
import os
import numpy as np
import pandas as pd
from torchvision.utils import make_grid
import torch


def train_epoch(loader, frame, metrics_opts):
    """Train model for one epoch

    This makes one pass through a dataloader and updates the model in the
    associated frame.

    :param loader: A pytorch DataLoader containing x,y pairs
      with which to train the model.
    :type loader: torch.data.utils.DataLoader
    :param frame: A Framework object wrapping both the model and the
      optimization setup.
    :type frame: Framework
    :param metrics_opts: A dictionary whose keys specify which metrics to
      compute on the predictions from the model.
    :type metrics_opts: dict
    :return (train_loss, metrics): A tuple containing the average epoch loss
      and the metrics on the training set.
    """
    loss, metrics = 0, {}
    frame.model.train()
    for x, y in loader:
        y_hat, _loss = frame.optimize(x, y)
        loss += _loss

        y_hat = frame.segment(y_hat)
        metrics_ = frame.metrics(y_hat, y, metrics_opts)

        update_metrics(metrics, metrics_)

    agg_metrics(metrics)

    return loss / len(loader.dataset), metrics


def validate(loader, frame, metrics_opts):
    """Compute Metrics on a Validation Loader

    To honestly evaluate a model, we should compute its metrics on a validation
    dataset. This runs the model in frame over the data in loader, compute all
    the metrics specified in metrics_opts.

    :param loader: A DataLoader containing x,y pairs with which to validate the
      model.
    :type loader: torch.utils.data.DataLoader
    :param frame: A Framework object wrapping both the model and the
      optimization setup.
    :type frame: Framework
    :param metrics_opts: A dictionary whose keys specify which metrics to
      compute on the predictions from the model.
    :type metrics_opts: dict
    :return (val_loss, metrics): A tuple containing the average validation loss
      and the metrics on the validation set.
    """
    loss, metrics = 0, {}
    channel_first = lambda x: x.permute(0, 3, 1, 2)
    frame.model.eval()
    for x, y in loader:
        with torch.no_grad():
            y_hat = frame.infer(x)
            loss += frame.calc_loss(channel_first(y_hat), channel_first(y)).item()

            y_hat = frame.segment(y_hat)
            metrics_ = frame.metrics(y_hat, y, metrics_opts)

            update_metrics(metrics, metrics_)

    agg_metrics(metrics)
    return loss / len(loader.dataset), metrics


def log_batch(epoch, n_epochs, i, n, loss, batch_size):
    """Helper to log a training batch

    :param epoch: Current epoch
    :type epoch: int
    :param n_epochs: Total number of epochs
    :type n_epochs: int
    :param i: Current batch index
    :type i: int
    :param n: total number of samples
    :type n: int
    :param loss: current epoch loss
    :type loss: float
    :param batch_size: training batch size
    :type batch_size: int
    """
    print(
        f"Epoch {epoch}/{n_epochs}, Training batch {i+1} of {int(n) // batch_size}, Loss = {loss/batch_size:.5f}",
        end="\r",
        flush=True,
    )


def log_metrics(writer, metrics, avg_loss, epoch, stage="train", mask_names=None):
    """Log metrics for tensorboard

    A function that logs metrics from training and testing to tensorboard

    Args:
        writer(SummaryWriter): The tensorboard summary object
        metrics(Dict): Dictionary of metrics to record
        avg_loss(float): The average loss across all epochs
        epoch(int): Total number of training cycles
        stage(String): Train/Val
        mask_names(List): Names of the mask(prediction) to log mmetrics for
    """
    writer.add_scalar(f"Loss/{stage}", avg_loss, epoch)
    if mask_names is None:
        mask_nums = len(list(metrics.values())[0])
        mask_names = [str(i) for i in range(mask_nums)]
    for k, v in metrics.items():
        for name, metric in zip(mask_names, v):
            writer.add_scalar(f"{name}_{str(k)}/{stage}", metric, epoch)


def log_images(writer, frame, batch, epoch, stage="train"):
    """Log images for tensorboard

    Args:
        writer (SummaryWriter): The tensorboard summary object
        frame (Framework): The model to use for inference
        batch (tensor): The batch of samples on which to make predictions
        epoch (int): Current epoch number
        stage (string): specified pipeline stage

    Return:
        Images Logged onto tensorboard
    """
    pm = lambda x: x.permute(0, 3, 1, 2)
    squash = lambda x: (x - x.min()) / (x.max() - x.min())

    x, y = batch

    y_hat = frame.act(frame.infer(x))
    y = torch.flatten(y.permute(0, 1, 3, 2), start_dim=2)
    y_hat = torch.flatten(y_hat.permute(0, 1, 3, 2), start_dim=2)

    if epoch == 0:
        writer.add_image(f"{stage}/x", make_grid(pm(squash(x[:, :, :, :3]))), epoch)
        writer.add_image(f"{stage}/y", make_grid(y.unsqueeze(1)), epoch)
    writer.add_image(f"{stage}/y_hat", make_grid(y_hat.unsqueeze(1)), epoch)

def update_metrics(main_metrics, batch_metrics):
    """Append --inplace-- a dict of tensors to another element by element
       Args:
            main_metrics (dict(troch.Tensor)): the matrix to append to
            batch_metrics (dict(troch.Tensor)): the new value to append to main_metrics
    """
    for k, v in batch_metrics.items():
        if k in main_metrics:
            main_metrics[k] = torch.cat((main_metrics[k], v.unsqueeze(-1)), 1)
        else:
            main_metrics[k] = v.unsqueeze(-1)

def agg_metrics(metrics):
    """Aggregate --inplace-- the mean of a matrix of tensor (across rows)
       Args:
            metrics (dict(troch.Tensor)): the matrix to get mean of"""
    for k, v in metrics.items():
        metrics[k] = metrics[k].mean(1)
        metrics[k] = metrics[k].data.cpu()
