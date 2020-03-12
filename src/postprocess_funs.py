#!/usr/bin/env python
from addict import Dict
from joblib import Parallel, delayed
from pathlib import Path
from shutil import copy
import addict
import json
import numpy as np
import os
import pandas as pd
import random


def filter_directory(slice_meta, filter_perc=0.2, filter_channel=0, n_cpu=10):
    """
    Return Paths for Pairs passing Filter Criteria

    :param filter_perc: The minimum percentage 1's in the filter_channel needed
      to pass the filter.
    :param filter_channel: The channel to do the filtering on.
    """
    img_paths, mask_paths = slice_meta["img_slice"].values, slice_meta["mask_slice"].values

    def wrapper(i):
        if i % 10 == 0:
            print(f"{i}/{len(img_paths)}")

        cur_ids = []
        mask = np.load(mask_paths[i])
        perc = mask[:, :, filter_channel].mean()
        if perc >= filter_perc:
            cur_ids.append({"img": img_paths[i], "mask": mask_paths[i]})

        return cur_ids

    para = Parallel(n_jobs=n_cpu)
    keep_ids = para(delayed(wrapper)(i) for i in range(len(mask_paths)))
    return sum(keep_ids, [])


def random_split(ids, split_ratio, **kwargs):
    random.shuffle(ids)
    sizes = len(ids) * np.array(split_ratio)
    ix = [int(s) for s in np.cumsum(sizes)]
    return {
        "train": ids[:ix[0]],
        "dev": ids[ix[0]:ix[1]],
        "test": ids[ix[1]:ix[2]]
    }


def reshuffle(split_ids, output_dir="output/"):
    for split_type in split_ids:
        path = Path(output_dir, split_type)
        os.makedirs(path, exist_ok=True)

    target_locs = {}
    for split_type in split_ids:
        n_ids = len(split_ids[split_type])
        target_locs[split_type] = n_ids * [{}]

        for i in range(n_ids):
            for im_type in ["img", "mask"]:
                source = split_ids[split_type][i][im_type]
                target = Path(output_dir, split_type, os.path.basename(source))
                copy(source, target)
                target_locs[split_type][i][im_type] = target

    return target_locs


def generate_stats(image_paths, sample_size, outpath="stats.json"):
    """
    Function to generate statistics of the input image channels

    :param image_paths: List of Paths to images in directory
    :param sample_size: int giving the size of the sample from which to compute the statistics
    :param outpath: str The path to the output json file containing computed statistics

    :return Dictionary with keys for means and stds across the channels in input images
    """
    sample_size = min(sample_size, len(image_paths))
    image_paths = np.random.choice(image_paths, sample_size, replace=False)
    images = [np.load(image_path) for image_path in image_paths]
    batch = np.stack(images)
    means = np.nanmean(batch, axis=(0,1,2))
    stds = np.nanstd(batch, axis=(0,1,2))

    with open(outpath, "w") as f:
        stats = {
            "means": means.tolist(),
            "stds": stds.tolist()
        }

        json.dump(stats,f)
    return(stats)


def normalize_(img, means, stds, **kwargs):
    """
        :param img: Input image to normalize
        :param means: Computed mean of the input channels
        :param stds: Computed standard deviation of the input channels

        :return img: Normalized img
    """
    for i in range(img.shape[2]):
        img[:,:,i] -= means[i]
        if stds[i] > 0:
            img[:,:,i] /= stds[i]
        else:
            img[:,:,i] = 0

    return img


def normalize(img, mask, stats, **kwargs):
    """wrapper for postprocess"""
    img = normalize_(img, stats["means"], stats["stds"])
    return img, mask


def postprocess(img_path, mask_path, funs_seq, **kwargs):
    """process a single image / mask pair"""
    img, mask = np.load(img_path), np.load(mask_path)
    for f in funs_seq:
        img, mask = f(img, mask, **kwargs)

    return img, mask
