#!/usr/bin/env python
from pathlib import Path
import numpy as np
import src.metrics
import src.models.unet
import torch
# from torch import optim

class Framework():

    def __init__(self,loss_fn=None, model_opts=None, optimizer_opts=None, metrics_opts=None, out_dir="outputs"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.out_dir = out_dir
        self.loss = None

        if loss_fn is None:
            loss_fn = torch.nn.BCEWithLogitsLoss()
        self.loss_fn = loss_fn.to(self.device)

        model_def = getattr(src.models.unet, model_opts.name)
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
        for k,v in self.metrics_opts.items():
            yhat_temp = self.y_hat
            if "threshold" in v.keys():
                yhat_temp = self.y_hat > v["threshold"]
            metric_fun = getattr(src.metrics,k)
            metric_value = metric_fun(yhat_temp,self.y.to(self.device))
            results.append(metric_value)

        return np.array(results)