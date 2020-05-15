#!/usr/bin/env python
from pathlib import Path
import numpy as np
import src.metrics
import src.models.unet
import torch

class Framework():

    def __init__(self,loss_fn=None, model_opts=None, optimizer_opts=None, metrics_opts=None, out_dir=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.out_dir = "outputs"

        if loss_fn is None:
            loss_fn = torch.nn.BCEWithLogitsLoss()
        self.loss_fn = loss_fn.to(self.device)

        model_def = getattr(src.unet, model_opts.name)
        self.model = model_def(**model_opts.args).to(self.device)


        optimizer_def = getattr(torch.optim, optimizer_opts.name)
        self.optimizer = optimizer_def(self.model.parameters(), **optimizer_opts.args)
        self.metrics_opts = metrics_opts

    def set_input(self, x, y):
        self.x = x.permute(0, 3, 1, 2).to(self.device)
        self.y = y.permute(0, 1, 2).to(self.device)

    def optimize(self):
        self.optimizer.zero_grad()
        self.y_hat = self.model(self.x)
        loss = self.loss(self.y_hat,self.y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save(self, out_dir, epoch):
        model_path = Path(self.out_dir, f"model_{epoch}.pt")
        optim_path = Path(self.out_dir, f"optim_{epoch}.pt")
        torch.save(self.model.state_dict(), model_path)
        torch.save(self.optimizer.state_dict(), optim_path)

    def infer(self, x):
        x = x.permute(0, 3, 1, 2).to(self.device)
        with torch.no_grad():
            return self.model(x)

    def loss(self, y, y_hat):
        return self.loss_fn(y, y_hat)


    def calculate_metrics(self):
        results = []
        for metrics in self.metrics_opts:
            for k,v in self.metrics_opts.items():
                yhat_temp = self.y_hat
                if "threshold" in v.keys():
                   yhat_temp = self.y_hat > v["threshold"]
                metric_fun = getattr(src.metrics,k)
                metric_value = metric_fun(yhat_temp,self.y.to(self.device))
                results.append(metric_value)

        return np.array(results)
