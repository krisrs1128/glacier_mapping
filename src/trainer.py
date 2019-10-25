#!/usr/bin/env python
import os
from pathlib import Path
import numpy as np
import torch
import comet_ml


class Trainer:
  def __init__(self, exp, model, config, train_data, dev_data, test_data):
    self.exp = exp
    self.model = model
    self.config = config
    self.train_data = train_data
    self.dev_data = dev_data
    self.test_data = test_data
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  def train(self):
    """
    Train Across Epochs
    """
    self.model.to(self.device)
    op = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
    loss_f = torch.nn.BCEWithLogitsLoss()

    for epoch in range(self.config.epochs):
      epoch_losses, mean_loss = self.train_epoch(op, loss_f)
      dev_loss = self.evaluate(loss_f)
      self.exp.log_metrics({
        "loss/train": mean_loss,
        "loss/dev": dev_loss
      }, step=epoch)

      if (epoch % self.config.save_freq) == 0:
        save_path = pathlib.Path(self.config.save_dir, f"model_{epoch}.pt")
        torch.save(self.model.state_dict(), save_path)


  def train_epoch(self, op, loss_f):
    """
    Train a Single Epoch
    """
    self.model.train()
    epoch_losses = []

    for img, mask in self.train_data:
      op.zero_grad()

      img, mask = img.to(self.device), mask.to(self.device)
      pred = self.model(img)
      loss = loss_f(pred, mask)
      epoch_losses.append(loss.item())
      loss.backward()
      op.step()

    return epoch_losses, np.mean(epoch_losses)


  def evaluate(self, loss_f, metric_f=None, mode="dev"):
      epoch_loss = 0
      epoch_metric = 0
      self.model.eval()

      if mode == "test":
        data = self.test_data
      else:
        data = self.dev_data

      for img, mask in data:
          with torch.no_grad():
              img, mask = img.to(self.device), mask.to(self.device)
              pred = self.model(img)
              loss = loss_f(pred, mask)
              epoch_loss += loss.item()
              if not metric_f:
                _, binary_pred = Trainer.get_pred_mask(pred)
                metric = metric_f(binary_pred, mask)
                epoch_metric += metric

      if not metric_f:
        return epoch_loss / len(data), epoch_metric / len(data)
      else:
        return epoch_loss / len(data)


  def predict(self, data, thresh=0.5):
    with torch.no_grad():
      pred = self.model(data)
      pred, binary_pred = Trainer.get_pred_mask(pred, thresh=thresh)
      return pred, binary_pred

  @staticmethod
  def get_pred_mask(pred, thresh=0.5):
    act = torch.nn.Sigmoid()
    pred = act(pred)
    binary_pred = pred.clone().detach()
    binary_pred[binary_pred >= thresh] = 1
    binary_pred[binary_pred < thresh] = 0

    return pred, binary_pred
