#!/usr/bin/env python
from collections import defaultdict
import numpy as np
import pathlib
import src.metrics as mtr
import src.utils
import torch
import wandb

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

    metrics = {'pixel_acc': mtr.pixel_acc, 'per': mtr.precision, 'recall':
               mtr.recall, 'dice': mtr.dice, 'iou': mtr.IoU}

    for epoch in range(self.config.n_epochs):
      self.train_epoch(op, loss_f)

      dev_loss, dev_metrics = self.evaluate(loss_f, metrics)
      train_loss, train_metrics = self.evaluate(loss_f, metrics, mode='train')

      wandb.log({"loss/train": train_loss, "loss/dev": dev_loss}, step=epoch)
      wandb.log({f'{k}/train': v for k, v in train_metrics.items()}, step=epoch)
      wandb.log({f'{k}/dev': v for k, v in dev_metrics.items()}, step=epoch)
      print(f"epoch {epoch}/{self.config.n_epochs}\ttrain loss: {mean_loss}\tdev loss: {dev_loss}")

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
    self.model.eval()

    data = get_attr(self, f"{mode}_data")
    epoch_metrics = defaultdict(int)
    wandb_imgs = []

    for i, (img, mask) in enumerate(data):
      with torch.no_grad():
        img, mask = img.to(self.device), mask.to(self.device)
        pred = self.model(img)
        loss = loss_f(pred, mask)
        epoch_loss += loss.item()

        epoch_metrics = utils.update_metrics(
          epoch_metrics,
          metric_fs,
          self.config.multi_class
        )
        if self.config.store_images and i % 10  == 0:
          wandb_images.append(utils.merged_image(img, mask, pred))

    wandb.log({f"{mode}_images": wandb_imgs})
    return (epoch_loss / len(data)), {name: value/len(data) for name, value in epoch_metrics.items()}

  def predict(self, data, thresh=0.5):
    """Given an image segment it."""
    with torch.no_grad():
      pred = self.model(data)
      act = matching_act(self.config.mluti_class)
      return utils.get_pred_mask(pred, act=act, thresh=thresh)
