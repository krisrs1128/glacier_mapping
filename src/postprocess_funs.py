import pandas as pd
import numpy as np
import addict
from addict import Dict
from pathlib import Path
import pdb
import glob
import os
from argparse import ArgumentParser


def filter_directory(slice_meta, filter_perc=0.2, filter_channel=0):
    """
    Return Paths for Pairs passing Filter Criteria

    :param filter_perc: The minimum percentage 1's in the filter_channel needed
      to pass the filter.
    :param filter_channel: The channel to do the filtering on.
    """
    keep_ids = []

    img_paths, mask_paths = slice_meta["img_slice"].values, slice_meta["mask_slices"].values
    for i, mask_path in enumerate(mask_paths):
        mask = np.load(mask_path)

        if i % 10 == 0:
            print(f"{i}/{len(img_paths)}")
        perc = mask[:, :, filter_channel].mean()

        if perc > filter_perc:
            keep_ids.append({
                "img": img_paths[i],
                "mask": mask_path
            })

    return keep_ids

def random_split(ids, split_ratio, **kwargs):
    ids = random.shuffle(ids)
    sizes = len(ids) * np.array(split_ratio)
    ix = np.cumsum(sizes)
    return {
        "train": ids[:ix[0]],
        "dev": ids[ix[0]:ix[1]],
        "test": ids[ix[1]:ix[2]]
    }

def reshuffle(split_ids, out_dir=None):
    if not out_dir:
        out_dir = "output/"

    for split_type in split_ids:
        path = Path(out_dir, split_type)
        os.mkdirs(path)

    target_locs = []
    for split_type in split_ids:
        n_ids = len(split_ids[split_type])
        target_locs.append({split_type: n_ids * [{}]})

        for i in range(n_ids):
            for im_type in ["img", "mask"]:
                source = split_ids[split_type][i][im_type]
                target = Path(out_dir, os.path.basename(source))
                os.replace(source, target)
                target_locs[split_type][i][im_type] = target

    return target_locs



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-d", "--slice_dir", type=str, help="path to directory with all the slices")
    parser.add_argument("-o", "--output_dir", type=str, help="path to output directory for postprocessed files")
    args = parser.parse_args()

    opts = Dict({"filter": True, "filter_percentage": 0, "filter_channel":0})
    mask_paths = glob.glob(args.slice_dir +"*mask*")

    for path in mask_paths:
        mask = np.load(path)


    keep_ids = filter_directory(input_dir, filter_perc=0.2, filter_channel=0)
