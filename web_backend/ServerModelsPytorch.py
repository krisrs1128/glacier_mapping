#!/usr/bin/env python3
from ServerModelsAbstract import BackendModel
from pathlib import Path
import numpy as np
import os
import time
import torch
import sys
sys.path.append(str(Path(os.environ["REPO_DIR"])))
from src.models.unet import Unet


class PytorchUNet(BackendModel):
    def __init__(self, model_spec, gpuid, verbose=False):
        self.input_size = model_spec["inputShape"]
        self.downweight_padding = 0
        self.stride_x, self.stride_y, _ = self.input_size

        if torch.cuda.is_available():
            state = torch.load(model_spec["fn"])
        else:
            state = torch.load(model_spec["fn"], map_location=torch.device("cpu"))

        self.model = Unet(**model_spec["args"])
        self.model.load_state_dict(state())
        self.model.eval()
        self.verbose = verbose

    def run(self, input_data, extent, on_tile=False):
        """
        makes predictions given
        """
        return self.run_model_on_tile(input_data)

    def run_model_on_tile(self, img, batch_size=32):
        """ Expects naip_tile to have shape (height, width, channels) and have values in the [0, 1] range.
        """
        height, width, _ = img.shape
        img = np.nan_to_num(img)
        for k in range(img.shape[2]):
            img[:, :, k] -= img[:, :, k].mean()
            img[:, :, k] /= (0.001 + img[:, :, k].std())

        counts = np.zeros((height, width), dtype=np.float32) + 0.000000001
        kernel = np.ones((self.input_size[0], self.input_size[1]), dtype=np.float32) * 0.1
        kernel[10:-10, 10:-10] = 1
        kernel[self.downweight_padding:self.downweight_padding+self.stride_y,
               self.downweight_padding:self.downweight_padding+self.stride_x] = 5

        batch, batch_indices = [], []
        batch_count = 0

        for y_index in (list(range(0, height - self.input_size[0], self.stride_y)) + [height - self.input_size[0],]):
            for x_index in (list(range(0, width - self.input_size[1], self.stride_x)) + [width - self.input_size[1],]):
                window = img[y_index:y_index+self.input_size[0], x_index:x_index+self.input_size[1], :]
                batch.append(window)
                batch_indices.append((y_index, x_index))
                batch_count += 1

        batch = np.transpose(np.array(batch), (0, 3, 1, 2)) # batch, channel, height, width
        batch = batch[:, :10, :, :] # temporary hack, to match channels

        with torch.no_grad():
            y_hat = self.model(torch.from_numpy(batch))
            y_hat = torch.nn.Sigmoid()(y_hat)
            y_hat = y_hat.detach().numpy()

        output = np.zeros((height, width), dtype=np.float32)
        for i, (y, x) in enumerate(batch_indices):
            output[y:y+self.input_size[0], x:x+self.input_size[1]] += y_hat[i] * kernel
            counts[y:y+self.input_size[0], x:x+self.input_size[1]] += kernel

        return (output / counts)[:, :, np.newaxis]

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
