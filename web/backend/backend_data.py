#!/usr/bin/env python
"""
Utilities for Managing VRTs

2020-05-05 18:11:08
"""
import glob
import pathlib
import subprocess
import argparse
import rasterio


def reproject_directory(input_dir, output_dir, dst_epsg=4326):
    """
    Warp all Tiffs from one directory to 4326
    """
    inputs = pathlib.Path(input_dir).glob("*.tif*")
    for im_path in inputs:
        print(f"reprojecting {str(im_path)}")
        loaded_im = rasterio.open(im_path)
        output_path = pathlib.Path(output_dir, f"{im_path.stem}-warped.tiff")
        subprocess.call(["gdalwarp", "-s_srs", str(loaded_im.crs), "-t_srs",
                         f"EPSG:{dst_epsg}", str(im_path),
                         "-wo", "NUM_THREADS=ALL_CPUS", str(output_path)])


def subset_channels(input_dir, output_dir, channels=[5, 4, 2]):
    ch_str = "".join([str(s) for s in channels])
    inputs = pathlib.Path(input_dir).glob("*.tif*")
    ch_list = sum([["-b", str(s)] for s in channels], [])

    for im_path in inputs:
        print(f"subsetting channels for {str(im_path)}")
        loaded_im = rasterio.open(im_path)
        output_path = pathlib.Path(output_dir, f"{im_path.stem}-{ch_str}.tiff")
        subprocess.call(
            ["gdal_translate", im_path, output_path] +
            ch_list +
            ["-a_nodata", "0"]
        )


def vrt_from_dir(input_dir, output_path="./output.vrt", **kwargs):
    """
    Build a VRT Indexing all Tiffs in a directory
    """
    inputs = [f for f in input_dir.glob("*.tif*")]
    subprocess.call(["gdalbuildvrt", "-o", output_path] + inputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge to a VRT")
    parser.add_argument("-d", "--input_dir", type=str)
    parser.add_argument("-o", "--output_dir", type=str, default="./")
    parser.add_argument("-n", "--output_name", type=str, default="output.vrt")
    parser.add_argument("-r", "--reproject", type=bool, default=False)
    parser.add_argument("-b", "--bandList", nargs="+", default=list(range(1, 16)))
    args = parser.parse_args()
    input_dir = pathlib.Path(args.input_dir)

    if args.reproject:
        warped_dir = input_dir / "warped"
        warped_dir.mkdir(exist_ok=True)
        reproject_directory(input_dir, warped_dir)
        input_dir = warped_dir

    if len(args.bandList) < 15:
        ch_str = "".join([str(s) for s in args.bandList])
        subset_dir = input_dir / "subset_channels" / ch_str
        subset_dir.mkdir(exist_ok=True)
        subset_channels(input_dir, subset_dir, args.bandList)
        input_dir = subset_dir

    vrt_path = pathlib.Path(args.output_dir, args.output_name)
    vrt_from_dir(input_dir, str(vrt_path))
