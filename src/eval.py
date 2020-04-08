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
import src.metrics
import src.mask
import geopandas as gpd

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


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

def get_hist(img, mask):
    """
    Defined:
        max number of points in csv for each label(n_points)
    Input:
        raster image, (expected raster image)
        mask (expected numpy array)
    Output:
        pandas dataframe
        graph
    """
    n_points = 1000
    df = pd.DataFrame(columns=["B1","B2","B3","B4","B5","B6_VCID_1","B6_VCID_2","B7","B8","BQA","elevation","slope","label"])

    # prepare the canvas
    x = np.arange(0.40, 2.40, 0.01)
    y = np.arange(0.40, 2.40, 0.01)
    x_values = np.array((0.49,0.56,0.67,0.84,1.66,2.22))
    fig, ax = plt.subplots()
    ax.axvspan(0.45, 0.52, alpha=0.2, color='blue')
    ax.axvspan(0.52, 0.60, alpha=0.2, color='green')
    ax.axvspan(0.63, 0.69, alpha=0.2, color='red')
    ax.axvspan(0.77, 0.90, alpha=0.2, color='grey')
    ax.axvspan(1.55, 1.75, alpha=0.2, color='grey')
    ax.axvspan(2.08, 2.35, alpha=0.2, color='grey')
    ax.set_xticks(x_values, minor=False)
    ax.set_xticklabels(["B1","B2","B3","B4","B5","B6_VCID_2"], fontdict=None, minor=False)
    plt.xlabel("Wavelength μm")
    plt.ylabel("Intensity")

    # read image
    img_np = img.read()
    img_np = np.transpose(img_np, (1, 2, 0))

    # img_np = img

    clean_index = np.argwhere(mask == 0)
    debris_index = np.argwhere(mask == 1)
    background_index = np.argwhere(mask == 2)
    np.random.shuffle(clean_index)
    np.random.shuffle(debris_index)
    np.random.shuffle(background_index)

    clean_index = clean_index[:n_points].tolist()
    debris_index = debris_index[:n_points].tolist()
    background_index = background_index[:n_points].tolist()


    clean_value = []
    debris_value = []
    background_value = []
    for index in clean_index:
        clean_value.append(img_np[index[0],index[1],:])
        row = pd.Series(list(np.append(img_np[index[0],index[1],:], "Clean Ice")), index=df.columns)
        df = df.append(row, ignore_index=True)
    for index in debris_index:
        debris_value.append(img_np[index[0],index[1],:])
        row = pd.Series(list(np.append(img_np[index[0],index[1],:], "Debris")), index=df.columns)
        df = df.append(row, ignore_index=True)
    for index in background_index:
        background_value.append(img_np[index[0],index[1],:])
        row = pd.Series(list(np.append(img_np[index[0],index[1],:], "Background")), index=df.columns)
        df = df.append(row, ignore_index=True)

    clean_value = np.asarray(clean_value)
    clean_mean = clean_value.mean(axis=0)
    clean_std = clean_value.std(axis=0)
    clean_mean = np.append(clean_mean[0:5],clean_mean[5])
    debris_value = np.asarray(debris_value)
    debris_mean = debris_value.mean(axis=0)
    debris_std = debris_value.std(axis=0)
    debris_mean = np.append(debris_mean[0:5],debris_mean[5])
    background_value = np.asarray(background_value)
    background_mean = background_value.mean(axis=0)
    background_std = background_value.std(axis=0)
    background_mean = np.append(background_mean[0:5],background_mean[5])

    for (x,y) in zip(x_values,clean_mean):
        ax.plot(x, y, 'bo')

    for (x,y) in zip(x_values,debris_mean):
        ax.plot(x, y, 'ro')

    for (x,y) in zip(x_values,background_mean):
        ax.plot(x, y, 'go')

    plt.plot(x_values, clean_mean, color='blue', label="Clean Glaciers")
    plt.plot(x_values, debris_mean, color='red', label="Debris Glaciers")
    plt.plot(x_values, background_mean, color='green', label="Background")
    plt.title("Wavelength vs Normalized intensity")
    ax.legend(bbox_to_anchor=(0.65, 1), loc='upper left', borderaxespad=0.)

    return df, plt


if __name__ == '__main__':
    print("loading raster")
    img = rasterio.open("/scratch/sankarak/data/glaciers/img_data/2005/hkh/LE07_140041_20051012.tif")

    print("getting mask")
    process_conf = "//home/sankarak/glacier_mapping/conf/postprocess.yaml"
    model_path = "/scratch/sankarak/data/glaciers/model_188.pt"
    state_dict = torch.load(model_path, map_location="cpu")
    model = Unet(10, 1, 4)
    model.load_state_dict(state_dict)
    y_hat = infer_tile(img, model, process_conf)
    y_hat = np.random.uniform(0, 1, (img.shape[0], img.shape[1], 2)) > 0.4

    print("getting mask")
    plt.imsave("prediction_mask.png", y_hat[:, :, 0]) # it's a one channel mask
    y_hat = torch.from_numpy(y_hat)

    # get true mask
    print("generating mask")
    mask = src.mask.generate_mask(img.meta, mask_shps)
    np.save("mask.npy", mask)

    mask_shps = [
        gpd.read_file("/scratch/sankarak/data/glaciers/vector_data/2005/hkh/data/Glacier_2005.shp"),
        gpd.read_file("/scratch/sankarak/data/glaciers/vector_data/2000/nepal/data/Glacier_2000.shp")
    ]
    mask_shps = [s.to_crs(img.meta["crs"].data) for s in mask_shps]

    print("loading mask")
    mask = np.load("mask.npy")
    # df, plt = get_hist(img, mask)

    # run metrics on mask, for each channel
    print("getting metrics")
    mask = torch.from_numpy(mask)
    metric_results = {}
    for k in range(mask.shape[2]):
        metric_results[k] = {}
        for metric in ["precision", "tp_fp_fn", "pixel_acc", "dice"]:
            l = getattr(src.metrics, metric)
            metric_results[k][metric] = l(mask[:, :, k], y_hat[:, :, k])

    # print / plot the result
    print(metric_results)
