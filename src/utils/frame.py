#!/usr/bin/env python
from pathlib import Path
import numpy as np
import src.utils.metrics
import src.models.unet
import src.models.unet_dropout
import torch
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Framework:
    def __init__(
        self,
        loss_fn=None,
        model_opts=None,
        optimizer_opts=None,
        metrics_opts=None,
        reg_opts=None,
        out_dir="outputs",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        self.out_dir = out_dir
        self.loss = None
        if loss_fn is None:
            loss_fn = torch.nn.BCEWithLogitsLoss()
        self.loss_fn = loss_fn.to(self.device)
        if model_opts.name == "Unet":
            model_def = getattr(src.models.unet, model_opts.name)
        elif model_opts.name == "UnetDropout":
            model_def = getattr(src.models.unet_dropout, model_opts.name)
        else:
            raise ValueError("Unknown model name")
        self.model = model_def(**model_opts.args).to(self.device)
        optimizer_def = getattr(torch.optim, optimizer_opts.name)
        self.optimizer = optimizer_def(self.model.parameters(), **optimizer_opts.args)
        self.lrscheduler = ReduceLROnPlateau(
            self.optimizer, "min", verbose=True, patience=500, min_lr=1e-6
        )
        self.metrics_opts = metrics_opts
        try:
            self.l1_lambda = reg_opts.l1_reg
        except:
            self.l1_lambda = False
        try:
            self.l2_lambda = reg_opts.l2_reg
        except:
            self.l2_lambda = False

    def set_input(self, x, y):
        self.x = x.permute(0, 3, 1, 2).to(self.device)
        self.y = y.permute(0, 3, 1, 2).to(self.device)

    def optimize(self):
        self.optimizer.zero_grad()
        self.y_hat = self.model(self.x)
        self.loss = self.calc_loss(self.y_hat, self.y)
        self.loss.backward()
        self.optimizer.step()
        return self.loss.item()

    def val_operations(self, val_loss):
        self.lrscheduler.step(val_loss)

    def save(self, out_dir, epoch):
        model_path = Path(self.out_dir, f"model_{epoch}.pt")
        optim_path = Path(self.out_dir, f"optim_{epoch}.pt")
        torch.save(self.model.state_dict(), model_path)
        torch.save(self.optimizer.state_dict(), optim_path)

    def infer(self, x):
        x = x.permute(0, 3, 1, 2).to(self.device)
        with torch.no_grad():
            return self.model(x).permute(0, 3, 2, 1)

    def calc_loss(self, y_hat, y):
        loss = self.loss_fn(y_hat, y)
        if self.l1_lambda:
            l1_regularization = torch.tensor(0.0).to(self.device)
            for param in self.model.parameters():
                l1_regularization += torch.sum(abs(param))
            loss = loss + self.l1_lambda * l1_regularization
        if self.l2_lambda:
            l2_regularization = torch.tensor(0.0).to(self.device)
            for param in self.model.parameters():
                l2_regularization += torch.norm(param, 2) ** 2
            loss = loss + self.l2_lambda * l2_regularization
        return loss

    def calculate_metrics(self):
        results = []
        for k, v in self.metrics_opts.items():
            b_metric = []
            for batch_y, batch_y_hat in zip(self.y, self.y_hat):
                c_metric = []
                for channel_wise_y, channel_wise_y_hat in zip(batch_y, batch_y_hat):
                    y = channel_wise_y.bool().to(self.device)
                    if "threshold" in v.keys():
                        y_hat = torch.sigmoid(channel_wise_y_hat) > v["threshold"]
                    else:
                        y_hat = channel_wise_y_hat.bool()
                    y_hat = y_hat.to(self.device)
                    metric_fun = getattr(src.utils.metrics, k)
                    metric_value = metric_fun(y_hat, y)
                    c_metric.append(metric_value)
                b_metric.append(np.mean(np.asarray(c_metric)))
            results.append(np.sum(np.asarray(b_metric)))
        return np.array(results)