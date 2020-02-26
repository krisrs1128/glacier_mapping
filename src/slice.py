#!/usr/bin/env python
"""
Convert Large Tiff and Mask files to Slices (512 x 512 subtiles)

2020-02-26 10:36:48
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def slice_tile(img, size=(512, 512, 12), overlap=6):
    """Slice an image into overlapping patches
    Args:
        img (np.array): image to be sliced
        size tuple(int, int, int): size of the slices
        overlap (int): how much the slices should overlap
    Returns:
        list of slices [np.array]"""
    patches = view_as_windows(img, size, step=size[0] - overlap)
    result = []
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            result.append(patches[i, j, 0])

    return result


def slice_pair(img, mask, **kwargs):
    img_slices = slice_tile(img, **kwargs)
    mask_slices = slice_tile(mask, **kwargs)
    return img_slices, mask_slices


def write_pair_slices(img_path, mask_path, out_dir=None, out_base=None,
                      metadata_path="metadata.csv", **kwargs):
    img = rasterio.open(img_path).read()
    mask = np.load(mask_path)
    img_slices, mask_slices = slice_pair(img, mask, **kwargs)

    # loop over slices for this tile / mask pair
    for k, img in enumerate(img_slices):
        img_slice_path = pathlib.Path(out_dir, f"{out_base}_img_{k}.npy")
        mask_slice_path = pathlib.Path(out_dir, f"{out_base}_mask_{k}.npy")

        # save outputs
        np.save(img, img_path)
        np.save(mask_slices[k], mask_path)
        metadata.append({
            "img_source": img_path,
            "mask_source": mask_path,
            "img_slice": img_slice_path,
            "mask_slice": mask_slice_path,
        })

    metadata = pd.DataFrame(metadata)
    metadata.to_csv(metadata_path, mode="a", index=False)
    return metadata

