#!/usr/bin/env python
import torch
import src.unet


class Framework():

    def __init__(self,loss_fn=None, model_opts=None, optimizer_opts=None):
        if loss_fn is None:
            loss_fn = torch.nn.BCEWithLogitsLoss()
        self.loss_fn = loss_fn
        model_def = getattr(src.unet, model_opts.name)
        self.model = model_def(**model_opts.args)

        optimizer_def = getattr(torch.optim, optimizer_opts.name)
        self.optimizer = optimizer_def(self.model.parameters(), **optimizer_opts.args)

    def set_input(self, x, y):
        self.x = x.permute(0, 3, 1, 2)
        self.y = y.permute(0, 1, 2)

    def optimize(self):
        self.optimizer.zero_grad()
        y_hat = self.model(self.x)
        loss = self.loss(y_hat, self.y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save(self, out_dir, epoch):
        model_path = Path(out_dir, f"model_{epoch}.pt")
        optim_path = Path(out_dir, f"optim_{epoch}.pt")
        torch.save(self.model.state_dict, model_path)
        torch.save(self.optimizer.state_dict, optim_path)

    def infer(self, x):
        x = x.permute(0, 3, 1, 2)
        with torch.no_grad():
            return self.model(x)

    def loss(self, y, y_hat):
        return self.loss_fn(y, y_hat)
