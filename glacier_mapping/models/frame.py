#!/usr/bin/env python
"""
Frame to Combine Model with Optimizer

This wraps the model and optimizer objects needed in training, so that each
training step can be concisely called with a single method (optimize).
"""
from pathlib import Path
import os
import torch
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
import metrics
import reg
import unet
import unet_dropout


class Framework:
    """
    Class to Wrap Training Steps
    """

    def __init__(self, loss_fn=None, model_opts=None, optimizer_opts=None,
                 reg_opts=None,):
        """
        Set Class Attrributes
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if loss_fn is None:
            loss_fn = torch.nn.BCEWithLogitsLoss()
        self.loss_fn = loss_fn.to(self.device)

        if model_opts.name == "Unet":
            model_def = getattr(unet, model_opts.name)
        elif model_opts.name == "UnetDropout":
            model_def = getattr(unet_dropout, model_opts.name)
        else:
            raise ValueError("Unknown model name")

        self.model = model_def(**model_opts.args).to(self.device)
        optimizer_def = getattr(torch.optim, optimizer_opts.name)
        self.optimizer = optimizer_def(self.model.parameters(), **optimizer_opts.args)
        self.lrscheduler = ReduceLROnPlateau(self.optimizer, "min",
                                             verbose=True, patience=500,
                                             min_lr=1e-6)
        self.reg_opts = reg_opts


    def optimize(self, x, y):
        """
        Take a single gradient step
        """
        x = x.permute(0, 3, 1, 2).to(self.device)
        y = y.permute(0, 3, 1, 2).to(self.device)

        self.optimizer.zero_grad()
        y_hat = self.model(x)
        loss = self.calc_loss(y_hat, y)
        loss.backward()
        self.optimizer.step()
        return y_hat.permute(0, 2, 3, 1), loss.item()

    def val_operations(self, val_loss):
        """
        Update the LR Scheduler
        """
        self.lrscheduler.step(val_loss)

    def save(self, out_dir, epoch):
        """
        Save a model checkpoint
        """
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        model_path = Path(out_dir, f"model_{epoch}.pt")
        optim_path = Path(out_dir, f"optim_{epoch}.pt")
        torch.save(self.model.state_dict(), model_path)
        torch.save(self.optimizer.state_dict(), optim_path)

    def infer(self, x):
        """
        Make a prediction for a given x
        """
        x = x.permute(0, 3, 1, 2).to(self.device)
        with torch.no_grad():
            return self.model(x).permute(0, 2, 3, 1)

    def calc_loss(self, y_hat, y):
        """
        Compute loss given a prediction
        """
        y_hat = y_hat.to(self.device)
        y = y.to(self.device)
        loss = self.loss_fn(y_hat, y)
        for reg_type in self.reg_opts.keys():
            reg_fun = getattr(reg, reg_type)
            penalty = reg_fun(
                self.model.parameters(),
                self.reg_opts[reg_type],
                self.device
            )
            loss += penalty

        return loss


    def metrics(self, y_hat, y, metrics_opts):
        """
        Loop over metrics in train.yaml
        """
        results = {}
        for k, metric in metrics_opts.items():
            if "threshold" in metric.keys():
                y_hat = y_hat > metric["threshold"]

                metric_fun = getattr(metrics, k)
                metric_value = metric_fun(y_hat, y)
            results[k] = np.mean(np.asarray(metric_value))
        return results
