#!/usr/bin/env python
from pathlib import Path
import numpy as np
import src.metrics
import src.models.unet
import src.models.unet_dropout
import torch
import os
# from torch import optim

class Framework():

    def __init__(self,loss_fn=None, model_opts=None, optimizer_opts=None, metrics_opts=None, out_dir="outputs"):
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
        # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', patience=25)
        self.metrics_opts = metrics_opts

    def set_input(self, x, y):
        self.x = x.permute(0, 3, 1, 2).to(self.device)
        self.y = y.permute(0, 3, 1, 2).to(self.device)

    def optimize(self):
        self.optimizer.zero_grad()
        self.y_hat = self.model(self.x)
        self.loss = self.calc_loss(self.y_hat,self.y)
        self.loss.backward()
        self.optimizer.step()
        return self.loss.item()

    def save(self, out_dir, epoch):
        model_path = Path(self.out_dir, f"model_{epoch}.pt")
        optim_path = Path(self.out_dir, f"optim_{epoch}.pt")
        torch.save(self.model.state_dict, model_path)
        torch.save(self.optimizer.state_dict, optim_path)

    def infer(self, x):
        x = x.permute(0, 3, 1, 2).to(self.device)
        with torch.no_grad():
            return self.model(x).permute(0, 3, 2, 1)

    def calc_loss(self, y, y_hat):
        return self.loss_fn(y, y_hat)


    def calculate_metrics(self):
        results = []
        for k, v in self.metrics_opts.items():
            b_metric = []
            for batch_y, batch_y_hat in zip(self.y, self.y_hat):
                c_metric = []
                for channel_wise_y, channel_wise_y_hat in zip(batch_y, batch_y_hat):
                    y = channel_wise_y.bool().to(self.device)
                    if "threshold" in v.keys():
                        y_hat = channel_wise_y_hat > v["threshold"]
                    else:
                        y_hat = channel_wise_y_hat.bool()
                    y_hat = y_hat.to(self.device)
                    metric_fun = getattr(src.metrics, k)
                    metric_value = metric_fun(y_hat, y)
                    c_metric.append(metric_value)
                b_metric.append(np.mean(np.asarray(c_metric)))
            results.append(np.sum(np.asarray(b_metric)))
        return np.array(results)