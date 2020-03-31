#!/usr/bin/env python
import numpy as np
import torch
from src.slice import slice_tile
import rasterio
from src.postprocess_funs import postprocess_tile
import yaml

"""
infer_tile(tile) -> mask

inputs:
    - A raster object giving the tile on which to do inference
    - a loaded model
      * infer(): given numpy patch of certain size, produce a numpy array of class predictions
      *
    - The size of the patches on which the model can do inference
    - preprocessing specification. Path to the preprocessing conf that was used.
 outputs:
    - Segmentation mask: Numpy array of same dimension as the input tile. K channels representing K class outputs.

inference_wrapper(VRT + selected region) -> mask for that region
"""

def merge_patches(files, overlap):
    """
    This function is to merge the patches in files with overlap = overlap
    2*3 != 3*2
    How to solve this? more information might be needed regarding the size of the final segmentation patch.
    """
    pass


def infer_tile(img, model, preprocess_conf, slice_opts=None):
    if slice_opts is None:
        slice_opts = {
            "overlap": 0,
            "size": (512, 512)
        }
    # To improve : overlap is hardcoded in src function rather than defining in configuration file.
    process_opts = yaml.load(open(preprocess_conf, "r"))

    slice_imgs = slice_tile(img.read(), **slice_opts)
    predictions = []
    for patch in slice_imgs:
        patch = postprocess_tile(patch, process_opts)
        predictions.append(model.infer(patch))

    return merge_patches(predictions, process_opts.data.overlap)
