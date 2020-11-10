#!/usr/bin/env python
"""
Utilities for Managing VRTs

2020-05-05 18:11:08
"""
import glob
import pathlib
import subprocess
import argparse
import gdal2tiles
import rasterio
from osgeo import gdal


def reproject_directory(input_dir, output_dir, dst_epsg=4326):
    """
    Warp all Tiffs from one directory to 4326
    """
    inputs = pathlib.Path(input_dir).glob("*.tif*")
    for im_path in inputs:
        print(f"reprojecting {str(im_path)}")
        loaded_im = rasterio.open(im_path)
        output_path = pathlib.Path(output_dir, f"{im_path.stem}-warped.tif")
        subprocess.call(["gdalwarp", "-s_srs", str(loaded_im.crs), "-t_srs",
                         f"EPSG:{dst_epsg}", str(im_path),
                         "-wo", "NUM_THREADS=ALL_CPUS", str(output_path)])


def vrt_from_dir(input_dir, output_path="./output.vrt", **kwargs):
    """
    Build a VRT Indexing all Tiffs in a directory
    """
    inputs = glob.glob(f"{input_dir}*.tif*")
    vrt_opts = gdal.BuildVRTOptions(**kwargs)
    gdal.BuildVRT(output_path, inputs, options=vrt_opts)


def tiles(input_vrt, output_dir, zoom_levels="8-14"):
    """
    Generate PNG tiles from a VRT
    """
    path = pathlib.Path(input_vrt)
    intermediate = f"{path.parent}/{path.resolve().stem}-byte.vrt"
    subprocess.call(["gdal_translate", "-ot", "Byte", input_vrt, f'{intermediate}'])
    gdal2tiles.generate_tiles(
        f"{intermediate}",
        output_dir,
        zoom=zoom_levels,
        verbose=True,
        tile_size=1056
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge to a VRT")
    parser.add_argument("-d", "--input_dir", type=str)
    parser.add_argument("-o", "--output_dir", type=str, default="./")
    parser.add_argument("-n", "--output_name", type=str, default="output.vrt")
    parser.add_argument("-t", "--tile", default=False)
    parser.add_argument("-b", "--bandList", nargs="+", default=list(range(1, 15)))
    parser.add_argument("-z", "--zoomLevels", nargs="+", default="8-14")
    args = parser.parse_args()

    reproject_directory(args.input_dir, args.output_dir)
    vrt_path = pathlib.Path(args.output_dir, args.output_name)
    vrt_from_dir(args.output_dir, str(vrt_path), bandList=args.bandList)
    if args.tile:
        tiles(vrt_path, args.output_dir, args.zoomLevels)
