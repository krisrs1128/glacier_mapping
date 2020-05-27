#!/usr/bin/env python
"""
Convert Large Tiff and Mask files to Slices (512 x 512 subtiles)

2020-02-26 10:36:48
"""
from geopandas.geodataframe import GeoDataFrame
from joblib import Parallel, delayed
from pathlib import Path
from skimage.util.shape import view_as_windows
import argparse
import numpy as np
import os
import pandas as pd
import rasterio
import shapely.geometry
from tqdm import tqdm


def slice_tile(img, size=(512, 512), overlap=6):
    """Slice an image into overlapping patches
    Args:
        img (np.array): image to be sliced
        size tuple(int, int, int): size of the slices
        overlap (int): how much the slices should overlap
    Returns:
        list of slices [np.array]"""
    size_ = (size[0], size[1], img.shape[2])
    patches = view_as_windows(img, size_, step=size[0] - overlap)
    result = []
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            result.append(patches[i, j, 0])
    return result


def slices_metadata(imgf, img_path, mask_path, size=(512, 512), overlap=6):
    meta = slice_polys(imgf, size, overlap)
    meta["img_source"] = img_path
    meta["mask_source"] = mask_path
    return meta


def slice_polys(imgf, size=(512, 512), overlap=6):
    """
    Get Polygons Corresponding to Slices
    """
    ix_row = np.arange(0, imgf.meta["height"], size[0] - overlap)
    ix_col = np.arange(0, imgf.meta["width"], size[1] - overlap)
    lats = np.linspace(imgf.bounds.bottom, imgf.bounds.top, imgf.meta["height"])
    longs = np.linspace(imgf.bounds.left, imgf.bounds.right, imgf.meta["width"])

    polys = []
    for i in range(len(ix_row) - 1):
        for j in range(len(ix_col) - 1):
            box = shapely.geometry.box(
                longs[ix_col[j]],
                lats[ix_row[i]],
                longs[ix_col[j + 1]],
                lats[ix_row[i + 1]],
            )
            polys.append(box)

    return GeoDataFrame(geometry=polys, crs=imgf.meta["crs"].to_string())


def slice_pair(img, mask, **kwargs):
    img_slices = slice_tile(img, **kwargs)
    mask_slices = slice_tile(mask, **kwargs)
    return img_slices, mask_slices


def write_pair_slices(
    img_path, mask_path, out_dir, out_base="slice", n_cpu=5, **kwargs
):
    imgf = rasterio.open(img_path)
    img = imgf.read().transpose(1, 2, 0)
    mask = np.load(mask_path)
    img_slices, mask_slices = slice_pair(img, mask, **kwargs)
    metadata = slices_metadata(imgf, img_path, mask_path, **kwargs)

    # loop over slices for individual tile / mask pairs
    slice_stats = []
    for k in tqdm(range(len(img_slices))):
        img_slice_path = Path(out_dir, f"{out_base}_img_{k:03}.npy")
        mask_slice_path = Path(out_dir, f"{out_base}_mask_{k:03}.npy")
        np.save(img_slice_path, img_slices[k])
        np.save(mask_slice_path, mask_slices[k])

        # update metadata
        stats = {"img_slice": str(img_slice_path), "mask_slice": str(mask_slice_path)}
        img_slice_mean = np.nan_to_num(img_slices[k].mean())
        mask_mean = mask_slices[k].mean(axis=(0, 1))
        stats.update({f"mask_mean_{i}": v for i, v in enumerate(mask_mean)})
        stats.update({f"img_mean": img_slice_mean})
        slice_stats.append(stats)

    slice_stats = pd.DataFrame(slice_stats)
    return pd.concat([metadata, slice_stats], axis=1)


if __name__ == "__main__":
    processed_dir = Path(os.environ["DATA_DIR"], "processed")

    parser = argparse.ArgumentParser(description="Slicing a single tiff / mask pair")
    parser.add_argument(
        "-m",
        "--mask_metadata",
        type=str,
        help="csv file mapping tiffs to masks.",
        default=processed_dir / "masks/mask_metadata.csv",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        help="directory to save all outputs",
        default=processed_dir / "slices/",
    )
    parser.add_argument(
        "-s",
        "--start_line",
        type=int,
        default=0,
        help="start line in the metadata, from which to start processing",
    )
    parser.add_argument(
        "-e",
        "--end_line",
        type=int,
        default=100,
        help="end line in the metadata, at which to stop processing",
    )
    parser.add_argument(
        "-b",
        "--out_base",
        type=str,
        help="Name to prepend to all the slices",
        default="slice",
    )
    parser.add_argument(
        "-c", "--n_cpu", type=int, help="number of CPU nodes to use", default=5
    )
    args = parser.parse_args()
    paths = pd.read_csv(args.mask_metadata)[args.start_line : args.end_line]

    # Slicing all the Tiffs in input csv file into specified output directory
    def wrapper(row):
        img_path = paths.iloc[row]["img"]
        mask_path = paths.iloc[row]["mask"]
        print(f"## Slicing tiff {row +1}/{len(paths)} ...")
        return write_pair_slices(
            img_path, mask_path, args.output_dir, f"slice_{paths.index[row]}"
        )

    Path(args.output_dir).mkdir(parents=True)
    para = Parallel(n_jobs=args.n_cpu)
    metadata = para(delayed(wrapper)(k) for k in range(len(paths)))
    metadata = pd.concat(metadata, axis=0)
    out_path = Path(
        args.output_dir, f"slices_{args.start_line}-{args.end_line}.geojson"
    )
    metadata.to_file(out_path, index=False, driver="GeoJSON")
