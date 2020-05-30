#!/usr/bin/env python
from addict import Dict
from pathlib import Path
from skimage.util.shape import view_as_windows
from src.utils.frame import Framework
from src.models.unet import Unet
from src.data.process_slices_funs import postprocess_tile
from torchvision.utils import save_image
import argparse
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import rasterio
import src.data.mask
import torch
import yaml


def squash(x):
    return (x - x.min()) / x.ptp()


def append_name(s, args):
    return f"{s}_{Path(args.input).stem}-{Path(args.model).stem}-{Path(args.process_conf).stem}.png"


def merge_patches(patches, overlap, subset_ix):
    I, J, _, height, width, channels = patches.shape
    result = np.zeros((I * height, J * width, channels))
    for i in range(I):
        for j in range(J):
            ix_i = i * (height - overlap)
            ix_j = j * (width - overlap)
            result[ix_i : (ix_i + height), ix_j : (ix_j + width)] = patches[i, j]

    return result[subset_ix[0], subset_ix[1]]


def inference(img, model, process_conf):
    """
    inference(tile) -> mask

    :param img: A (unprocessed) numpy array on which to do inference.
    :param model: A pytorch model on which to perform inference. We assume it
      can accept images of size specified in process_conf.slice.size.
    :param process_conf: The path to a yaml file giving the postprocessing
      options. Used to convert the raw tile into the tensor used for inference.
    :return prediction: A segmentation mask of the same width and height as img.
    """
    process_opts = Dict(yaml.safe_load(open(process_conf, "r")))
    slice_opts = process_opts.slice
    channels = process_opts.process_funs.extract_channel.img_channels

    # reshape, pad, and slice the input
    img = np.transpose(img, (1, 2, 0))
    size_ = (slice_opts.size[0], slice_opts.size[1], img.shape[2])
    subset_ix = (
        slice(size_[0], img.shape[0] + size_[0]),
        slice(size_[1], img.shape[1] + size_[1]),
    )
    img_pad = np.zeros(
        (img.shape[0] + 2 * size_[0], img.shape[1] + 2 * size_[1], img.shape[2])
    )

    img_pad[subset_ix[0], subset_ix[1]] = img
    slice_imgs = view_as_windows(img_pad, size_, step=size_[0] - slice_opts.overlap)

    I, J, _, _, _, _ = slice_imgs.shape
    predictions = np.zeros((I, J, 1, size_[0], size_[1], 1))
    patches = np.zeros((I, J, 1, size_[0], size_[1], len(channels)))

    for i in range(I):
        for j in range(J):
            patch, _ = postprocess_tile(slice_imgs[i, j, 0], process_opts.process_funs)

            patches[i, j, :] = patch
            patch = np.transpose(patch, (2, 0, 1))
            patch = torch.from_numpy(patch).float().unsqueeze(0)

            with torch.no_grad():
                y_hat = model(patch).numpy()
                predictions[i, j, 0] = np.transpose(y_hat, (0, 2, 3, 1))

    x = merge_patches(patches, process_opts.slice.overlap, subset_ix)
    y_hat = merge_patches(predictions, process_opts.slice.overlap, subset_ix)
    return x, y_hat


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
    df = pd.DataFrame(
        columns=[
            "B1",
            "B2",
            "B3",
            "B4",
            "B5",
            "B6_VCID_1",
            "B6_VCID_2",
            "B7",
            "B8",
            "BQA",
            "elevation",
            "slope",
            "label",
        ]
    )

    # prepare the canvas
    x = np.arange(0.40, 2.40, 0.01)
    y = np.arange(0.40, 2.40, 0.01)
    x_values = np.array((0.49, 0.56, 0.67, 0.84, 1.66, 2.22))
    fig, ax = plt.subplots()
    ax.axvspan(0.45, 0.52, alpha=0.2, color="blue")
    ax.axvspan(0.52, 0.60, alpha=0.2, color="green")
    ax.axvspan(0.63, 0.69, alpha=0.2, color="red")
    ax.axvspan(0.77, 0.90, alpha=0.2, color="grey")
    ax.axvspan(1.55, 1.75, alpha=0.2, color="grey")
    ax.axvspan(2.08, 2.35, alpha=0.2, color="grey")
    ax.set_xticks(x_values, minor=False)
    ax.set_xticklabels(
        ["B1", "B2", "B3", "B4", "B5", "B6_VCID_2"], fontdict=None, minor=False
    )
    plt.xlabel("Wavelength μm")
    plt.ylabel("Intensity")

    # read image
    img_np = img.read()
    img_np = np.transpose(img_np, (1, 2, 0))

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
        clean_value.append(img_np[index[0], index[1], :])
        row = pd.Series(
            list(np.append(img_np[index[0], index[1], :], "Clean Ice")),
            index=df.columns,
        )
        df = df.append(row, ignore_index=True)
    for index in debris_index:
        debris_value.append(img_np[index[0], index[1], :])
        row = pd.Series(
            list(np.append(img_np[index[0], index[1], :], "Debris")), index=df.columns
        )
        df = df.append(row, ignore_index=True)
    for index in background_index:
        background_value.append(img_np[index[0], index[1], :])
        row = pd.Series(
            list(np.append(img_np[index[0], index[1], :], "Background")),
            index=df.columns,
        )
        df = df.append(row, ignore_index=True)

    clean_value = np.asarray(clean_value)
    clean_mean = clean_value.mean(axis=0)
    clean_std = clean_value.std(axis=0)
    clean_mean = np.append(clean_mean[0:5], clean_mean[5])
    debris_value = np.asarray(debris_value)
    debris_mean = debris_value.mean(axis=0)
    debris_std = debris_value.std(axis=0)
    debris_mean = np.append(debris_mean[0:5], debris_mean[5])
    background_value = np.asarray(background_value)
    background_mean = background_value.mean(axis=0)
    background_std = background_value.std(axis=0)
    background_mean = np.append(background_mean[0:5], background_mean[5])

    for (x, y) in zip(x_values, clean_mean):
        ax.plot(x, y, "bo")

    for (x, y) in zip(x_values, debris_mean):
        ax.plot(x, y, "ro")

    for (x, y) in zip(x_values, background_mean):
        ax.plot(x, y, "go")

    plt.plot(x_values, clean_mean, color="blue", label="Clean Glaciers")
    plt.plot(x_values, debris_mean, color="red", label="Debris Glaciers")
    plt.plot(x_values, background_mean, color="green", label="Background")
    plt.title("Wavelength vs Normalized intensity")
    ax.legend(bbox_to_anchor=(0.65, 1), loc="upper left", borderaxespad=0.0)

    return df, plt


if __name__ == "__main__":
    data_dir = Path(os.environ["DATA_DIR"])
    root_dir = Path(os.environ["ROOT_DIR"])

    parser = argparse.ArgumentParser(description="Draw inferences from a raw tiff")
    parser.add_argument(
        "-m",
        "--model",
        default=data_dir / "runs/minimal_run/models/model_5.pt",
        help="path to the model to use for predictions",
    )
    parser.add_argument(
        "-i",
        "--input",
        default=data_dir / "raw/img_data/2010/nepal/Nepal_139041_20111225.tif",
        help="path to tiff file to draw inference on",
    )
    parser.add_argument(
        "-t",
        "--train_conf",
        default=root_dir / "conf/train.yaml",
        help="path to the configuration file used for training (to fetch model initialization parameters)",
    )
    parser.add_argument(
        "-p",
        "--process_conf",
        default=root_dir / "conf/postprocess.yaml",
        help="path to the slice processing file",
    )
    parser.add_argument(
        "-c", "--channels", default=[2, 4, 5], help="input channels to save for png"
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        default=data_dir / "outputs",
        help="input channels to save for png",
    )
    args = parser.parse_args()

    print("loading raster")
    img = rasterio.open(args.input).read()
    train_conf = Dict(yaml.safe_load(open(args.train_conf, "r")))
    process_conf = Dict(yaml.safe_load(open(args.process_conf, "r")))

    model = Framework(
        model_opts=train_conf.model_opts, optimizer_opts=train_conf.optim_opts
    ).model

    if torch.cuda.is_available():
        state_dict = torch.load(args.model)
    else:
        state_dict = torch.load(args.model, map_location="cpu")

    model.load_state_dict(state_dict)
    print("making predictions")
    x, y_hat = inference(img, model, args.process_conf)

    # convert input to png
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.imsave(
        output_dir / append_name("input", args),
        squash(img[args.channels].transpose(1, 2, 0)),
    )

    plt.imsave(output_dir / append_name("x", args), squash(x[:, :, :3]))

    # convert prediction to png
    for j in range(y_hat.shape[2]):
        plt.imsave(
            output_dir / append_name(f"y_hat-prepred-{j}", args), squash(y_hat[:, :, j])
        )

