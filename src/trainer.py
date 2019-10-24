#!/usr/bin/env python
import os
import logging

import numpy as np

import torch

class Config:
  def __init__(self, lr, epochs, save_dir, save_freq=1):
    self.lr = lr
    self.epochs = epochs
    self.save_dir = save_dir
    self.save_freq = save_freq

class Trainer:
  def __init__(self, model, config,
               train_data, dev_data, test_data,
               device):
    self.model = model
    self.config = config
    self.train_data = train_data
    self.dev_data = dev_data
    self.test_data = test_data
    self.device = device

  def train(self):
    op = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
    loss_f = torch.nn.BCEWithLogitsLoss()
    for epoch in range(self.config.epochs):
      epoch_losses, mean_loss = self.train_epoch(op, loss_f)
      dev_loss = self.evaluate(loss_f)
      logging.info('Epoch {}: training loss = {}, dev loss = {}'.format(
                    epoch, mean_loss, dev_loss))
      if (epoch % self.config.save_freq) == 0:
        save_path = os.path.join(self.config.save_dir,
                                 'model_{}.pt'.format(epoch))
        torch.save(self.model.state_dict(), save_path)



  def train_epoch(self, op, loss_f):
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

    mean_loss = sum(epoch_losses) / len(epoch_losses)
    return epoch_losses, mean_loss


  def evaluate(self, loss_f, metric_f=None, mode='dev'):
      epoch_loss = 0
      epoch_metric = 0
      self.model.eval()

      if mode == 'test':
        data = self.test_data
      else:
        data = self.dev_data
      n = len(data)
      for img, mask in data:
          with torch.no_grad():
              img, mask = img.to(self.device), mask.to(self.device)
              pred = self.model(img)
              loss = loss_f(pred, mask)
              epoch_loss += loss.item()
              if metric_f is not None:
                _, binary_pred = Trainer.get_pred_mask(pred)
                metric = metric_f(binary_pred, mask)
                epoch_metric += metric

      if metric_f is not None:
        return (epoch_loss / n), (epoch_metric / n)
      else:
        return (epoch_loss / n)

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
