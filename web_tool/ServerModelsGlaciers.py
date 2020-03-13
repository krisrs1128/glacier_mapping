#!/usr/bin/env python3
from web_tool.ServerModelsAbstract import BackendModel
import torch
import numpy as np
import torch.nn as nn
import os
from src.unet import Unet
from torch.autograd import Variable
import time


def softmax(output):
    output_max = np.max(output, axis=2, keepdims=True)
    exps = np.exp(output-output_max)
    exp_sums = np.sum(exps, axis=2, keepdims=True)
    return exps/exp_sums

class UnetFineTune(BackendModel):

    def __init__(self, model_fn, gpuid):

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuid)
        self.output_channels = 5
        self.input_size = 240
        self.did_correction = False
        self.model_fn = model_fn
        self.opts ## =  load yaml here
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.inf_framework = InferenceFramework(Unet, self.opts)
        self.inf_framework.load_model(self.model_fn)
        for param in self.inf_framework.model.parameters():
            param.requires_grad = False

        # ------------------------------------------------------
        # Step 2
        #   Pre-load augment model seed data
        # ------------------------------------------------------
        self.current_features = None

        self.augment_base_x_train = []
        self.augment_base_y_train = []

        self.augment_x_train = []
        self.augment_y_train = []
        self.model = self.inf_framework.model
        self.init_model()
        self.model_trained = False
        self.glacier_data = None
        self.correction_labels = None
        self.tile_padding = 0

        self.down_weight_padding = 40

        self.stride_x = self.input_size - self.down_weight_padding * 2
        self.stride_y = self.input_size - self.down_weight_padding * 2
        self.batch_x = []
        self.batch_y = []
        self.num_corrected_pixels = 0
        self.batch_count = 0
        self.run_done = False
        self.rows = 892
        self.cols = 892

    def run(self, glacier_data, naip_fn, extent, padding):
        if self.correction_labels is not None:
            self.set_corrections()

        # apply padding to the output_features
        x = glacier_data
        x = np.swapaxes(x, 0, 2)
        x = np.swapaxes(x, 1, 2)
        x = x[:4, :, :]
        glacier_data = x / 255.0
        output = self.run_model_on_tile(glacier_data)
        if padding > 0:
            self.tile_padding = padding
        self.glacier_data = glacier_data  # keep non-trimmed size, i.e. with padding
        self.correction_labels = np.zeros((glacier_data.shape[1], glacier_data.shape[2], self.output_channels),
                                          dtype=np.float32)
        self.last_output = output
        return output

    def set_corrections(self):
        self.did_correction = True
        num_labels = np.count_nonzero(self.correction_labels)
        batch_count = 0
        correction_labels = self.correction_labels

        batch_xi = self.glacier_data[:, 130:self.rows + 130, 130:self.cols + 130]
        batch_yi = np.argmax(correction_labels[130:self.rows + 130, 130:self.cols + 130, :], axis=2)
        if(num_labels>0):
            self.batch_x.append(batch_xi)
            self.batch_y.append(batch_yi)
            self.num_corrected_pixels += num_labels
            self.batch_count += batch_count

    def retrain(self, train_steps=25, learning_rate=0.0015):
        pass
