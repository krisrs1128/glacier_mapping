#!/usr/bin/env python
from rasterio.mask import mask as rasterio_mask
from addict import Dict
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import rasterio
import yaml

def crop_raster(raster_img, vector_data):
    vector_crs = rasterio.crs.CRS(vector_data.crs)
    if vector_crs != raster_img.meta['crs']:
        vector_data = vector_data.to_crs(raster_img.meta['crs'].data)

    mask = rasterio_mask(raster_img, list(vector_data.geometry), crop=False)[0]
    return mask


def get_mask(raster_img, vector_data, nan_value=0):
    # check if both have the same crs
    # follow the raster data as it's easier, faster
    # and doesn't involve saving huge new raster data
    vector_crs = rasterio.crs.CRS(vector_data.crs)
    if vector_crs != raster_img.meta['crs']:
        vector_data = vector_data.to_crs(raster_img.meta['crs'].data)

    mask = rasterio_mask(raster_img, list(vector_data.geometry), crop=False)[0]
    binary_mask = mask[0, :, :]
    binary_mask[np.isnan(binary_mask)] = nan_value
    binary_mask[binary_mask > 0] = 1

    return binary_mask


def slice_image(img, size=(512, 512)):
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
    plt.imshow(sat_rgb(sat_img))
    display_sat_bands(sat_img)

def sat_rgb(sat_img, indeces=(0, 1, 2)):
    rgb = np.stack([sat_img[:, :, indeces[0]],
                    sat_img[:, :, indeces[1]],
                    sat_img[:, :, indeces[2]]], 2).astype('int32')
    return rgb

def display_sat_bands(sat_img, bands=10, band_names=None, l7=True):
    cols = 5
    rows = int(bands/cols)
    fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(30, int(30/rows)))
    if l7:
        band_names = ['Blue', 'Green', 'Red', 'Near infrared',
                      'Shortwave infrared 1', 'Low-gain Thermal Infrared',
                      'High-gain Thermal Infrared',
                      'Shortwave infrared 2','Panchromatic','BQA']
    elif band_names is None:
        band_names = [str(i + 1) for i in range(bands)]

    for i in range(rows):
        for j in range(cols):
            ax[i, j].imshow(sat_img[:, :, i * cols + j])
            ax[i, j].set_title('{} band'.format(band_names[i * cols + j]), size=20)
    fig.tight_layout()


def display_sat_mask(sat_img, mask, borders=None):
    rows = 1
    cols = 3 if borders is not None else 2

    fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(30, int(30/rows)))

    ax[0].imshow(sat_rgb(sat_img))
    ax[0].title.set_text('RGB')
    ax[1].imshow(mask)
    ax[1].title.set_text('Binary Mask')
    if borders is not None:
        ax[2].imshow(borders)
        ax[2].title.set_text('Country Borders')

    fig.tight_layout()
    display_sat_bands(sat_img)


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
        conf_path = Path(__file__).parent.parent / "shared" / conf_name
        assert conf_path.exists()

    return merge_defaults({"model": {}, "train": {}, "data": {}}, conf_path)
