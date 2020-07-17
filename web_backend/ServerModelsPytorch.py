#!/usr/bin/env python3
from pathlib import Path
from src.infer import infer_conv
from src.models.unet import Unet
from web_backend.ServerModelsAbstract import BackendModel
import numpy as np
import os
import sys
import time
import torch
import yaml
import json

DATA_DIR = os.environ["DATA_DIR"]

class PytorchUNet(BackendModel):
    def __init__(self, model_spec, gpuid, verbose=False):
        self.input_size = model_spec["inputShape"]
        self.downweight_padding = 0
        self.stride_x, self.stride_y, _ = self.input_size
        self.process_conf = model_spec["process"]

        model_path = Path(DATA_DIR, model_spec["fn"])
        if torch.cuda.is_available():
            state = torch.load(model_path)
        else:
            state = torch.load(model_path, map_location=torch.device("cpu"))

        self.model = Unet(**model_spec["args"])
        self.model.load_state_dict(state)
        self.model.eval()
        self.verbose = verbose

    def run(self, img, **kwargs):
        opts = yaml.load(open(self.process_conf))
        channels = opts["process_funs"]["extract_channel"]["img_channels"]
        stats_path = opts["process_funs"]["normalize"]["stats_path"]
        stats = json.load(open(stats_path, "r"))
        means = np.array(stats["means"])
        stds = np.array(stats["stds"])
        return infer_conv(img, self.model, channels, means, stds)

    def run_model_on_batch(self, batch_data, batch_size=32, predict_central_pixel_only=False):
        """ Expects batch_data to have shape (none, 240, 240, 4) and have values in the [0, 255] range.
        """
        raise NotImplementedError("run_model_on_batch method of ServerModelsPytorch not implemented.")

    def retrain(self, **kwargs):
        raise NotImplementedError("retrain method of ServerModelsPytorch not implemented.")

    def add_sample(self, tdst_row, bdst_row, tdst_col, bdst_col, class_idx):
        raise NotImplementedError("add_sample method of ServerModelsPytorch not implemented.")

    def undo(self):
        raise NotImplementedError("undo method of ServerModelsPytorch not implemented.")

    def reset(self):
        raise NotImplementedError("reset method of ServerModelsPytorch not implemented.")
