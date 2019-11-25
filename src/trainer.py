import os
import logging
from collections import defaultdict

import numpy as np

import torch

from metrics import pixel_acc, dice, IoU, precision, recall, diceloss


class Config:
    def __init__(self, lr, epochs, save_dir, save_freq=1, multi_class=False):
        self.lr = lr
        self.epochs = epochs
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.multi_class = multi_class


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
        if self.config.multi_class:
            loss_f = torch.nn.CrossEntropyLoss()
        else:
            # loss_f = torch.nn.BCEWithLogitsLoss()
            loss_f = diceloss()
        
        metrics = {'pixel_acc': pixel_acc, 'per': precision, 'recall': recall,
                   'dice': dice, 'iou': IoU}
        for epoch in range(self.config.epochs):
            epoch_losses, mean_loss = self.train_epoch(op, loss_f)
            dev_loss, dev_metrics = self.evaluate(loss_f, metrics)
            train_loss, train_metrics = self.evaluate(
                loss_f, metrics, mode='train')
            logging.info('''Epoch {}: training loss = {}, train_metrics = {},
                      dev loss = {}, dev_metrics = {}'''.format(
                epoch, train_loss, train_metrics, dev_loss, dev_metrics))
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

    def evaluate(self, loss_f, metric_fs=None, mode='dev'):
        epoch_loss = 0
        epoch_metrics = defaultdict(int)
        self.model.eval()

        if mode == 'test':
            data = self.test_data
        elif mode == 'train':
            data = self.train_data
        else:
            data = self.dev_data
        n = len(data)
        for img, mask in data:
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

        if metric_fs is not None:
            return (epoch_loss / n), {name: value/n for name, value in epoch_metrics.items()}
        else:
            return (epoch_loss / n)

    def predict(self, data, thresh=0.5):
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
        pred = act(pred)
        binary_pred = pred.clone().detach()
        binary_pred[binary_pred >= thresh] = 1
        binary_pred[binary_pred < thresh] = 0

        return pred, binary_pred
