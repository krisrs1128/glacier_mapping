#!/usr/bin/env python
import numpy as np
import torch
from src.slice import slice_tile
from skimage import data
from skimage.util.shape import view_as_windows
import rasterio
from src.unet import Unet
from src.postprocess_funs import postprocess_tile
import yaml
from addict import Dict
import matplotlib.pyplot as plt
import cv2
"""
infer_tile(tile) -> mask

inputs:
    - A raster object giving the tile on which to do inference
    - a loaded model
      * infer(): given numpy patch of certain size, produce a numpy array of class predictions
      *
    - The size of the patches on which the model can do inference
    - processing specification. Path to the processing conf that was used.
 outputs:
    - Segmentation mask: Numpy array of same dimension as the input tile. K channels representing K class outputs.

inference_wrapper(VRT + selected region) -> mask for that region
"""

def merge_patches(patches, overlap, output_size):
    """
    This function is to merge the patches in files with overlap = overlap
    2*3 != 3*2
    How to solve this? more information might be needed regarding the size of the final segmentation patch.
    """
    I, J, _, height, width, channels = patches.shape
    result = np.zeros((I * height, J * width, channels))
    for i in range(I):
        for j in range(J):
            ix_i = i * (height - overlap)
            ix_j = j * (width - overlap)
            result[ix_i:(ix_i + height), ix_j:(ix_j + width)] = patches[i, j]

    return result[:output_size[0], :output_size[1]]


def infer_tile(img, model, process_conf):
    # To improve : overlap is hardcoded in src function rather than defining in configuration file.
    process_opts = Dict(yaml.load(open(process_conf, "r")))
    slice_opts = process_opts.slice

    # img_np = img.read()
    img_np = img
    size_ = (slice_opts.size[0], slice_opts.size[1], img_np.shape[2])
    img_pad = np.zeros((img_np.shape[0] + size_[0], img_np.shape[1] + size_[1], img_np.shape[2]))
    slice_imgs = view_as_windows(img_pad, size_, step=size_[0] - slice_opts.overlap)

    I, J, _ = img_np.shape
    predictions = torch.zeros(I, J)
    for i in range(I):
        for j in range(J):
            path = postprocess_tile(
                slice_imgs[i, j, 0],
                process_opts.process_funs
            )
            predictions[i, j] = model(patch)

    return merge_patches(predictions, process_opts.slice.overlap, im_np.shape)

if __name__ == '__main__':
    # img = rasterio.open("/Users/krissankaran/Desktop/LE07_140041_20051012.tif")
    img = np.random.uniform(size=(5000, 4000, 12))
    process_conf = "/Users/krissankaran/Desktop/glacier_mapping/conf/postprocess.yaml"
    model_path = "/Users/krissankaran/Downloads/model_188.pt"

    state_dict = torch.load(model_path, map_location="cpu")
    model = Unet(10, 1, 4)
    model = model.load_state_dict(state_dict)
    y_hat = infer_tile(img, model, process_conf)
    plt.imsave("prediction_mask.png", y_hat)
