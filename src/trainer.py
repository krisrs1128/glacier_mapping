#!/usr/bin/env python
import os
import logging
from collections import defaultdict

import numpy as np
import pathlib
import torch
import wandb
from src.metrics import pixel_acc, dice, IoU, precision, recall, diceloss

class Trainer:
  def __init__(self, model, config, train_data, dev_data, test_data):
    self.model = model
    self.config = config
    self.train_data = train_data
    self.dev_data = dev_data
    self.test_data = test_data
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  def train(self):
    """Train Across Epochs"""

    self.model.to(self.device)
    wandb.watch(self.model)
    op = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
    if self.config.multi_class:
      loss_f = torch.nn.CrossEntropyLoss()
    else:
      loss_f = diceloss()

    metrics = {'pixel_acc': pixel_acc, 'per': precision, 'recall': recall,
                   'dice': dice, 'iou': IoU}

    for epoch in range(self.config.n_epochs):
      self.train_epoch(op, loss_f)

      dev_loss, dev_metrics = self.evaluate(loss_f, metrics)
      train_loss, train_metrics = self.evaluate(loss_f, metrics, mode='train')

      print(f"epoch {epoch}/{self.config.n_epochs}\ttrain loss: {train_loss}\tdev loss: {dev_loss}")
      wandb.log({"loss/train": train_loss, "loss/dev": dev_loss}, step=epoch)
      wandb.log({f'{k}/train': v for k, v in train_metrics.items()}, step=epoch)
      wandb.log({f'{k}/dev': v for k, v in dev_metrics.items()}, step=epoch)

      if (epoch % self.config.save_freq) == 0:
        save_path = pathlib.Path(self.config.output_path, f"model_{epoch}.pt")
        torch.save(self.model.state_dict(), save_path)


  def train_epoch(self, op, loss_f):
    """Train a Single Epoch"""

    self.model.train()
    epoch_losses = []

    for i, (img, mask) in enumerate(self.train_data):
      op.zero_grad()

      img, mask = img.to(self.device), mask.to(self.device)
      pred = self.model(img)
      loss = loss_f(pred, mask)
      epoch_losses.append(loss.item())
      loss.backward()
      op.step()
      print(f"batch {i}/{len(self.train_data)}\tloss: {loss}", end="\r")

    return epoch_losses, np.mean(epoch_losses)


  def evaluate(self, loss_f, metric_fs={}, mode='dev'):
    """Evaluate a dataset and return loss and metrics."""

    epoch_loss = 0
    epoch_metrics = defaultdict(int)
    self.model.eval()

    if mode == 'test':
      data = self.test_data
    elif mode == 'train':
      data = self.train_data
    else:
      data = self.dev_data

    wandb_imgs = []
    for i, (img, mask) in enumerate(data):
      with torch.no_grad():
        img, mask = img.to(self.device), mask.to(self.device)
        pred = self.model(img)
        loss = loss_f(pred, mask)
        epoch_loss += loss.item()
        if metric_fs is not None:
          if self.config.multi_class:
            act = torch.nn.Softmax(dim=1)
          else:
            act = torch.nn.Sigmoid()
          _, binary_pred = Trainer.get_pred_mask(pred, act=act)
          for name, fn in metric_fs.items():
            metric = fn(binary_pred, mask)
            epoch_metrics[name] += metric

        if self.config.store_images and i % 10  == 0:
          img, pred, mask = img[:, :3].cpu(), pred.unsqueeze(1).cpu(), mask.unsqueeze(1).cpu()
          pred = act(pred)
          for j in range(img.shape[0]):
            merged = np.concatenate([img[j], pred[j, [0, 0, 0]], mask[j, [0, 0, 0]]], axis=1)
            merged = (merged.transpose(2, 1, 0) - np.min(merged))/ np.ptp(merged)
            wandb_imgs.append(wandb.Image(merged))

    wandb.log({f"{mode}_images": wandb_imgs})
    return (epoch_loss / len(data)), {name: value/len(data) for name, value in epoch_metrics.items()}

  def predict(self, data, thresh=0.5):
    """Given an image segment it."""

    with torch.no_grad():
      pred = self.model(data)
      if self.config.multi_class:
        act = torch.nn.Softmax(dim=1)
      else:
        act = torch.nn.Sigmoid()
      pred, binary_pred = Trainer.get_pred_mask(pred,
        act=act, thresh=thresh)
      return pred, binary_pred

  @staticmethod
  def get_pred_mask(pred, act=torch.nn.Sigmoid(), thresh=0.5):
    """Given the logits of a model predict a segmentation mask."""

    pred = act(pred)
    binary_pred = pred.clone().detach()
    binary_pred[binary_pred >= thresh] = 1
    binary_pred[binary_pred < thresh] = 0

    return pred, binary_pred
