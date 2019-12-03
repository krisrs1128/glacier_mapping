#!/usr/bin/env python
from addict import Dict
from rasterio.mask import mask as rasterio_mask
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import rasterio
import torch
import wandb
import yaml


def crop_raster(raster_img, vector_data):
    """Crop a raster image according to given vector data and
       return the cropped version as numpy array"""
    vector_crs = rasterio.crs.CRS(vector_data.crs)
    if vector_crs != raster_img.meta["crs"]:
        vector_data = vector_data.to_crs(raster_img.meta["crs"].data)

    mask = rasterio_mask(raster_img, list(vector_data.geometry), crop=False)[0]

    return mask

def get_snow_index(img, thresh=None):
    """Given a satelitte image return the snow index,
       default is cahnnels first landsat7 format"""
    # channels first
    index = np.zeros_like(img[0])
    # for division by zero errors
    mask = (img[1, :, :] + img[4, :, :]) != 0
    values = (img[1, :, :] - img[4, :, :]) / (img[1, :, :] + img[4, :, :])
    index[mask] = values[mask]

    if thresh is not None:
        return index > thresh

    return index

def get_debris_glaciers(img, mask, thresh=0.6):
    """Given an image and labels construct pseudo labels of the debris glaciers,
       as any labels doesn"t captured by snow index"""
    snow_i = np.array(get_snow_index(img, thresh=thresh))
    mask = np.array(mask)
    debris_mask = np.zeros_like(mask)
    debris_mask[(snow_i == 0) & (mask == 1)] = 1

    return debris_mask


def merge_mask_snow_i(img, mask, thresh=0.6):
    """Return multi-class of a binary class mask,
       using snow_index to construct pseudo labels."""
    snow_i = get_snow_index(img, thresh=thresh)
    hybrid_mask = np.zeros_like(mask)
    hybrid_mask[(snow_i == 1) & (mask == 1)] = 1
    hybrid_mask[(snow_i == 0) & (mask == 1)] = 2

    return hybrid_mask


def get_bg(mask):
    """"Adds extra channel to a mask to represent background,
       to make it one hot vector"""
    if len(mask.shape) == 2:
        fg = mask
    else:
        fg = np.logical_or.reduce(mask, axis=2)
    bg = np.logical_not(fg)

    return np.stack((fg, bg))


def get_mask(raster_img, vector_data, nan_value=0):
    """Get a mask from a raster for a given vector data.
    Args:
        raster_img (rasterio dataset object): the rater image to mask
        vector_data (iterable polygon data): the labels to mask according to
        nan_value (int): the value to fill nan areas with
    Returns:
        a binary mask (np.array)"""

    # check if both have the same crs
    # follow the raster data as its easier, faster
    # and doesn't involve saving huge new raster data
    vector_crs = rasterio.crs.CRS(vector_data.crs)
    if vector_crs != raster_img.meta["crs"]:
        vector_data = vector_data.to_crs(raster_img.meta["crs"].data)

    mask = rasterio_mask(raster_img, list(vector_data.geometry), crop=False)[0]
    binary_mask = mask[0, :, :]
    binary_mask[np.isnan(binary_mask)] = nan_value
    binary_mask[binary_mask > 0] = 1

    return binary_mask


def slice_image(img, size=(512, 512)):
    """Given an image slice according to size with no overlapping.
    Args:
        img (np.array): image to be sliced
        size tuple(int, int): size of the slice
    Returns:
        list of slices [np.array]"""
    if img.ndim == 2:
        h, w = img.shape
    else:
        h, w, _ = img.shape

    h_slices = math.ceil(h / size[0])
    w_slices = math.ceil(w / size[1])

    slices = []
    for i in range(h_slices):
        for j in range(w_slices):
            start_i, end_i = i * size[0], (i + 1) * size[0]
            start_j, end_j = j * size[1], (j + 1) * size[1]

            # the last tile need to be of the same size as well
            if end_i > h:
                start_i = -size[0]
            if end_j > w:
                start_j = -size[1]
            if img.ndim == 2:
                slices.append(img[start_i:end_i, start_j:end_j])
            else:
                slices.append(img[start_i:end_i, start_j:end_j, :])

    return slices


def display_sat_image(sat_img):
    """Display the RGB and bands of satelitte image"""
    plt.imshow(sat_rgb(sat_img))
    display_sat_bands(sat_img)


def sat_rgb(sat_img, indeces=(0, 1, 2), channel_first=False):
    """Given a 3d array return a 3 channels combination (default is rgb)
    Args:
        sat_img (nmupy.array): the image to use
        indeces (int, int, int): what are the channels of interest
        channels_first (bool): if sat_image has channels as last or first
    Returns:
        numpy.array: the three channels (rgb)"""

    if channel_first:
        sat_img = np.moveaxis(sat_img, 0, 2)
    rgb = np.stack([sat_img[:, :, indeces[0]],
                    sat_img[:, :, indeces[1]],
                    sat_img[:, :, indeces[2]]], 2).astype("int32")
    return rgb


def display_sat_bands(sat_img, bands=10, band_names=None, l7=True):
    """Given a satelitte image displat all bands
    Args:
        sat_img (numpy.array): the satelittle image to display
        bands (int): how many bands in the image
        band_names [str]: names of the bands
        l7 (bool): whether to use landsat7 band names """
    cols = 5
    rows = int(bands/cols)
    fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(30, int(30/rows)))
    if l7:
        band_names = ["Blue", "Green", "Red", "Near infrared",
                      "Shortwave infrared 1", "Low-gain Thermal Infrared",
                      "High-gain Thermal Infrared",
                      "Shortwave infrared 2", "Panchromatic", "BQA"]
    elif band_names is None:
        band_names = [str(i + 1) for i in range(bands)]

    for i in range(rows):
        for j in range(cols):
            ax[i, j].imshow(sat_img[:, :, i * cols + j])
            ax[i, j].set_title("{} band".format(
                band_names[i * cols + j]), size=20)
    fig.tight_layout()


def display_sat_mask(sat_img, mask, borders=None):
    """Given a satelitte image and mask display the RGB and the band beside the mask
    Args:
        sat_img (numpy.array): channels last image
        mask (numpy.array): a mask
        borders (numpy.array): an extra mask for the border """
    rows = 1
    cols = 3 if borders is not None else 2

    fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(30, int(30/rows)))

    ax[0].imshow(sat_rgb(sat_img))
    ax[0].title.set_text("RGB")
    ax[1].imshow(mask)
    ax[1].title.set_text("Binary Mask")
    if borders is not None:
        ax[2].imshow(borders)
        ax[2].title.set_text("Country Borders")

    fig.tight_layout()
    display_sat_bands(sat_img)


def sample_param(sample_dict):
    """sample a value (hyperparameter) from the instruction in the
    sample dict:
    {
        "sample": "range | list",
        "from": [min, max, step] | [v0, v1, v2 etc.]
    }
    if range, as np.arange is used, "from" MUST be a list, but may contain
    only 1 (=min) or 2 (min and max) values, not necessarily 3

    Args:
        sample_dict (dict): instructions to sample a value

    Returns:
        scalar: sampled value
    """
    if "sample" not in sample_dict:
        return sample_dict
    if sample_dict["sample"] == "range":
        value = np.random.choice(np.arange(*sample_dict["from"]))
    elif sample_dict["sample"] == "list":
        value = np.random.choice(sample_dict["from"])
    elif sample_dict["sample"] == "uniform":
        value = np.random.uniform(*sample_dict["from"])
    else:
        raise ValueError("Unknonw sample type in dict " + str(sample_dict))
    return value


def load_conf(path):
    path = pathlib.Path(path).resolve()
    print("Loading conf from", str(path))
    with open(path, "r") as stream:
        try:
            return Dict(yaml.safe_load(stream))
        except yaml.YAMLError as exc:
            print(exc)


def merge_defaults(extra_opts, conf_path):
    print("Loading params from", conf_path)
    result = load_conf(conf_path)
    for group in ["model", "train", "data"]:
        if group in extra_opts:
            for k, v in extra_opts[group].items():
                result[group][k] = v
    for group in ["model", "train", "data"]:
        for k, v in result[group].items():
            if isinstance(v, dict):
                v = sample_param(v)
            result[group][k] = v

    return Dict(result)


def get_opts(conf_path):
    if not pathlib.Path(conf_path).exists():
        conf_name = conf_path
        if not conf_name.endswith(".yaml"):
            conf_name += ".yaml"
        conf_path = pathlib.Path(__file__).parent.parent / "shared" / conf_name
        assert conf_path.exists()

    return merge_defaults({"model": {}, "train": {}, "data": {}}, conf_path)


def get_pred_mask(pred, act=torch.nn.Sigmoid(), thresh=0.5):
    """Given the logits of a model predict a segmentation mask."""
    pred = act(pred)
    binary_pred = pred.clone().detach() >= thresh
    return pred, binary_pred


def matching_act(multi_class=False):
    if multi_class:
        act = torch.nn.Softmax(dim=1)
    else:
        act = torch.nn.Sigmoid()
    return act


def update_metrics(epoch_metrics, pred, mask, metric_fs, multi_class=False):
    if metric_fs:
        act = matching_act(multi_class)
        _, binary_pred = get_pred_mask(pred, act=act)
        for name, fn in metric_fs.items():
            epoch_metrics[name] += fn(binary_pred, mask)

    return epoch_metrics


def merged_image(img, mask, pred, act):
    img, pred, mask = img[:, :3].cpu(), pred.unsqueeze(1).cpu(), mask.unsqueeze(1).cpu()
    pred = act(pred)
    result = []
    for j in range(img.shape[0]):
        merged = np.concatenate([img[j], pred[j, [0, 0, 0]], mask[j, [0, 0, 0]]], axis=1)
        merged = (merged.transpose(2, 1, 0) - np.min(merged))/ np.ptp(merged)
        result.append(wandb.Image(merged))

    return result
