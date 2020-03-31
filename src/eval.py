#!/usr/bin/env python
import numpy as np
import torch
from src.slice import slice_tile
from skimage import data
from skimage.util.shape import view_as_windows
# import rasterio
from src.postprocess_funs import postprocess_tile
import yaml
from addict import Dict
import matplotlib.pyplot as plt

"""
infer_tile(tile) -> mask

inputs:
    - A raster object giving the tile on which to do inference
    - a loaded model
      * infer(): given numpy patch of certain size, produce a numpy array of class predictions
      *
    - The size of the patches on which the model can do inference
    - postprocessing specification. Path to the postprocessing conf that was used.
 outputs:
    - Segmentation mask: Numpy array of same dimension as the input tile. K channels representing K class outputs.

inference_wrapper(VRT + selected region) -> mask for that region
"""

def merge_patches(patches, overlap):
    """
    This function is to merge the patches in files with overlap = overlap
    2*3 != 3*2
    How to solve this? more information might be needed regarding the size of the final segmentation patch.
    """
    I, J, _, height, width, channels = patches.shape

    result = np.zeros((I * (height - overlap) + overlap, J * (width - overlap) + overlap, channels))
    # overlap_count = np.zeros((I * (height - ...)))
    for i in range(I):
        for j in range(J):
            ix_i = i * (height - overlap)
            ix_j = j * (height - overlap)
            result[ix_i:(ix_i + height), ix_j:(ix_j + width)] = patches[i, j]
            # overlap_count[ix_i:(... )] += 1

    # pointwise divide patches / overlap_count
    return result


def infer_tile(img, model, postprocess_conf):
    # To improve : overlap is hardcoded in src function rather than defining in configuration file.
    process_opts = Dict(yaml.load(open(postprocess_conf, "r")))
    slice_opts = postprocess_opts.slice

    img_np = img.read()
    slice_imgs = view_as_windows(img_np, step=slice_opts.size[0] - slice_opts.overlap)

    I, J, _ = img_np.shape
    predictions = torch.zeros(I, J)
    for i in range(I):
        for j in range(J):
            path = postprocess_tile(slice_imgs[i, j], process_opts)
            predictions[i, j] = model.infer(patch)

    return merge_patches(predictions, process_opts.data.overlap)

if __name__ == '__main__':

    # img = rasterio.open("/Users/krissankaran/Desktop/LE07_140041_20051012.tif")
    img_np = data.retina()
    slice_imgs = view_as_windows(img_np, (100, 100, 3), step=100 - 6)
    merged = merge_patches(slice_imgs, 6)
    plt.imsave("original.png", img_np[:, :, :3] / 255)
    plt.imsave("merged.png", merged[:, :, :3] / 255)
