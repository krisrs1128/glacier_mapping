#!/usr/bin/env python
"""
Inference Module

This module contains functions for drawing predictions from an already trained
model, writing the results to file, and extracting corresponding geojsons.
"""
from pathlib import Path
from addict import Dict
import numpy as np
import rasterio
import torch
import yaml
import geopandas as gpd
import shapely.geometry
from shapely.ops import unary_union
import skimage.measure
from skimage.util.shape import view_as_windows
from rasterio.windows import Window
from .data.process_slices_funs import postprocess_tile
from .models.frame import Framework


def squash(x):
    return (x - x.min()) / x.ptp()


def append_name(s, args, filetype="png"):
    return f"{s}_{Path(args.input).stem}-{Path(args.model).stem}-{Path(args.process_conf).stem}.{filetype}"


def write_geotiff(y_hat, meta, output_path):
    """
    Write predictions to geotiff

    :param y_hat: A numpy array of predictions.
    :type y_hat: np.ndarray
    """
    # create empty raster with write geographic information
    dst_file = rasterio.open(
        output_path, 'w',
        driver='GTiff',
        height=y_hat.shape[0],
        width=y_hat.shape[1],
        count=y_hat.shape[2],
        dtype=np.float32,
        crs=meta["crs"],
        transform=meta["transform"]
    )

    y_hat = 255.0 * y_hat.astype(np.float32)
    for k in range(y_hat.shape[2]):
        dst_file.write(y_hat[:, :, k], k + 1)


def predict_tiff(path, model, subset_size=None, conf_path="conf/postprocess.yaml"):
    """
    Load a raster and make predictions on a subwindow
    """
    imgf = rasterio.open(path)
    if subset_size is not None:
        img = imgf.read(window=Window(0, 0, subset_size[0], subset_size[1]))
    else:
        img = imgf.read()
    x, y_hat = inference(img, model, conf_path)
    return img, x, y_hat


def merge_patches(patches, overlap):
    I, J, _, height, width, channels = patches.shape
    result = np.zeros((I * height, J * width, channels))
    for i in range(I):
        for j in range(J):
            ix_i = i * (height - overlap)
            ix_j = j * (width - overlap)
            result[ix_i : (ix_i + height), ix_j : (ix_j + width)] = patches[i, j]

    return result


def inference(img, model, process_conf, overlap=0, infer_size=1024, device=None):
    """Make predictions on an unprocessed tiff

    :param img: A (unprocessed) numpy array on which to do inference.
    :type img: np.array
    :param model: A pytorch model on which to perform inference. We assume it
      can accept images of size specified in process_conf.slice.size.
    :type model: pytorch.nn.Module
    :param process_conf: The path to a yaml file giving the postprocessing
      options. Used to convert the raw tile into the tensor used for inference.
    :type process_conf: string
    :param overlap: The number of overlapping pixels when splitting the patches
      on which to apply the model.
    :type overlap: int
    :param infer_size: The size of the square on which to perform inference.
    :type infer_size: int
    :param device: The device (gpu or cpu) on which to run inference.
    :type device: torch.device
    :return prediction: A segmentation mask of the same width and height as img.
    :type prediction: np.array
    """
    process_opts = Dict(yaml.safe_load(open(process_conf, "r")))
    channels = process_opts.process_funs.extract_channel.img_channels
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # reshape, pad, and slice the input
    size_ = img.shape
    img = pad_to_valid(img)
    img = np.transpose(img, (1, 2, 0))
    slice_size = (
        min(img.shape[0], infer_size),
        min(img.shape[1], infer_size),
        img.shape[2]
    )
    slice_imgs = view_as_windows(img, slice_size, step=slice_size[0] - overlap)

    I, J, _, _, _, _ = slice_imgs.shape
    predictions = np.zeros((I, J, 1, slice_size[0], slice_size[1], 1))
    patches = np.zeros((I, J, 1, slice_size[0], slice_size[1], len(channels)))

    for i in range(I):
        for j in range(J):
            patch, _ = postprocess_tile(slice_imgs[i, j, 0], process_opts.process_funs)
            patches[i, j, :] = patch
            patch = np.transpose(patch, (2, 0, 1))
            patch = torch.from_numpy(patch).float().unsqueeze(0)

            with torch.no_grad():
                patch = patch.to(device)
                y_hat = model(patch).cpu().numpy()
                y_hat = 1 / (1 + np.exp(-y_hat))
                predictions[i, j, 0] = np.transpose(y_hat, (0, 2, 3, 1))

    x = merge_patches(patches, overlap)
    y_hat = merge_patches(predictions, overlap)
    return x[:size_[1], :size_[2], :], y_hat[:size_[1], :size_[2], :]


def next_multiple(size):
    return np.ceil(size / 512) * 512


def pad_to_valid(img):
    size_ = img.shape
    out_rows = next_multiple(size_[1])
    out_cols = next_multiple(size_[2])

    pad_shape = (int(out_rows - size_[1]), int(out_cols - size_[2]))
    return np.pad(img, ((0, 0), (0, pad_shape[0]), (0, pad_shape[1])))


def convert_to_geojson(y_hat, bounds, threshold=0.8):
    """Convert a probability mask to geojson

    :param y_hat: A three dimensional numpy array of mask probabilities.
    :type y_hat: np.array
    :param bounds: The latitude / longitude bounding box of the region
      to write as geojson.
    :type bounds: tuple
    :param threshold: The probability above which an object is
      segmented into the geojson.
    :type threshold: float
    :return (geo_interface, geo_df) (tuple): tuple giving the geojson and
      geopandas data frame corresponding to the thresholded y_hat.
    """
    contours = skimage.measure.find_contours(y_hat, threshold, fully_connected="high")

    for i in range(len(contours)):
        contours[i] = contours[i][:, [1, 0]]
        contours[i][:, 1] = y_hat.shape[1] - contours[i][:, 1]
        contours[i][:, 0] = bounds[0] + (bounds[2] - bounds[0]) * contours[i][:, 0] / y_hat.shape[0]
        contours[i][:, 1] = bounds[1] + (bounds[3] - bounds[1]) * contours[i][:, 1] / y_hat.shape[1]

    contours = [c for c in contours if len(c) > 2]
    polys = [shapely.geometry.Polygon(a) for a in contours]
    polys = unary_union([p for p in polys if p.area > 4e-6])
    mpoly = shapely.geometry.multipolygon.MultiPolygon(polys)
    mpoly = mpoly.simplify(tolerance=0.0005)
    geo_df = gpd.GeoSeries(mpoly)
    return geo_df.__geo_interface__, geo_df


def load_model(train_yaml, model_path):
    """
    :param train_yaml: The path to the yaml file containing training options.
    :param model_path: The path to the saved model checkpoint, from which to
    load the state dict.
    :return model: The model with checkpoint loaded.
    """
    # loads an empty model, without weights
    train_conf = Dict(yaml.safe_load(open(train_yaml, "r")))
    model = Framework(torch.nn.BCEWithLogitsLoss(), train_conf.model_opts, train_conf.optim_opts).model

    # if GPU is available, inference will be faster
    if torch.cuda.is_available():
        state_dict = torch.load(model_path)
    else:
        state_dict = torch.load(model_path, map_location="cpu")

    model.load_state_dict(state_dict)
    return model
