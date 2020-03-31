#!/usr/bin/env python
from addict import Dict
from skimage.util.shape import view_as_windows
from src.postprocess_funs import postprocess_tile
from src.unet import Unet
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
import yaml


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
    """
    infer_tile(tile) -> mask

    :param img: A raster tile on which to do inference.
    :param model: A pytorch model on which to perform inference. We assume it
      can accept images of size specified in process_conf.slice.size.
    :param process_conf: The path to a yaml file giving the postprocessing
      options. Used to convert the raw tile into the tensor used for inference.
    :return prediction: A segmentation mask of the same width and height as img.
    """
    process_opts = Dict(yaml.load(open(process_conf, "r")))
    slice_opts = process_opts.slice

    # reshape and slice hte input
    img_np = img.read()
    img_np = np.transpose(img_np, (1, 2, 0))
    size_ = (slice_opts.size[0], slice_opts.size[1], img_np.shape[2])
    img_pad = np.zeros((img_np.shape[0] + size_[0], img_np.shape[1] + size_[1], img_np.shape[2]))
    slice_imgs = view_as_windows(img_pad, size_, step=size_[0] - slice_opts.overlap)

    I, J, _, _, _, _ = slice_imgs.shape
    predictions = np.zeros((I, J, 1, size_[0], size_[1], 1))
    for i in range(I):
        for j in range(J):
            patch, _ = postprocess_tile(
                slice_imgs[i, j, 0],
                process_opts.process_funs
            )
            patch = np.transpose(patch, (2, 0, 1))[None, :, :, :]
            patch = torch.from_numpy(patch).float()

            with torch.no_grad():
                y_hat = model(patch).numpy()
                predictions[i, j, 0] = np.transpose(y_hat, (1, 2, 0))

    return merge_patches(predictions, process_opts.slice.overlap, img_np.shape)

if __name__ == '__main__':
    img = rasterio.open("/scratch/sankarak/data/glaciers/img_data/2005/hkh/LE07_140041_20051012.tif")
    process_conf = "//home/sankarak/glacier_mapping/conf/postprocess.yaml"
    model_path = "/scratch/sankarak/data/glaciers/model_188.pt"

    state_dict = torch.load(model_path, map_location="cpu")
    model = Unet(10, 1, 4)
    model.load_state_dict(state_dict)
    y_hat = infer_tile(img, model, process_conf)
    plt.imsave("prediction_mask.png", y_hat[:, :, 0]) # it's a one channel mask
