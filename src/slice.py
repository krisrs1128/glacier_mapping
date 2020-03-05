#!/usr/bin/env python
"""
Convert Large Tiff and Mask files to Slices (512 x 512 subtiles)

2020-02-26 10:36:48
"""
from joblib import Parallel, delayed
from skimage.util.shape import view_as_windows
import argparse
import numpy as np
import os
import pandas as pd
import pathlib
import rasterio

def slice_tile(img, size=(512,512), overlap=6):
    """Slice an image into overlapping patches
    Args:
        img (np.array): image to be sliced
        size tuple(int, int, int): size of the slices
        overlap (int): how much the slices should overlap
    Returns:
        list of slices [np.array]"""
    size_ = (size[0], size[1], img.shape[2])
    patches = view_as_windows(img, size_, step=size[0]-overlap)
    result = []
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            result.append(patches[i,j,0])
    return result

def slice_pair(img, mask, **kwargs):
    img_slices = slice_tile(img, **kwargs)
    mask_slices = slice_tile(mask, **kwargs)
    return img_slices, mask_slices


def write_pair_slices(img_path, mask_path, out_dir=None, out_base="slice",
                      metadata_path="slice_metadata.csv", **kwargs):
    if out_dir is None:
        out_dir = os.getcwd()

    img = rasterio.open(img_path).read().transpose(1, 2, 0)
    mask = np.load(mask_path)
    img_slices, mask_slices = slice_pair(img, mask, **kwargs)
    metadata = []

    # loop over slices for this tile / mask pair
    for k, img in enumerate(img_slices):
        img_slice_path = pathlib.Path(out_dir, f"{out_base}_img_{k}.npy")
        mask_slice_path = pathlib.Path(out_dir, f"{out_base}_mask_{k}.npy")

        # save outputs
        np.save(img_slice_path, img)
        np.save(mask_slice_path, mask_slices[k])
        metadata.append({
            "img_source": img_path,
            "mask_source": mask_path,
            "img_slice": img_slice_path,
            "mask_slice": mask_slice_path,
        })

    metadata = pd.DataFrame(metadata)
    metadata.to_csv(pathlib.Path(out_dir, metadata_path), mode="a", index=False)
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Slicing a single tiff / mask pair")
    parser.add_argument("-p", "--paths_csv", type=str, help="csv file mapping tiffs to masks.", default="/scratch/sankarak/data/glaciers_azure/masks/metadata.csv")
    parser.add_argument("-o", "--output_dir", type=str, help="directory to save all outputs", default="/scratch/sankarak/data/glaciers_azure/slices/")
    parser.add_argument("-b", "--out_base", type=str, help="Name to prepend to all the slices", default="slice")
    parser.add_argument("-c", "--n_cpu", type=int, help="number of CPU nodes to use", default=5)
    args = parser.parse_args()

    paths = pd.read_csv(args.paths_csv)

    ## Slicing all the Tiffs in input csv file into specified output directory
    def wrapper(row):
        img_path=paths.iloc[row]["img"]
        mask_path=paths.iloc[row]["mask"]
        print(f"##Slicing tiff {row +1}/{len(paths)} ...")
        write_pair_slices(img_path, mask_path, args.output_dir, f"{args.out_base}_{row}")
    para = Parallel(n_jobs=args.n_cpu)
    para(delayed(wrapper)(k) for k in range(len(paths)))
