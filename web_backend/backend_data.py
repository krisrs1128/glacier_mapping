#!/usr/bin/env python
"""
Utilities for Managing VRTs

2020-05-05 18:11:08
"""
import numpy as np
import argparse
import pandas as pd
import fiona
import shapely.geometry
import pathlib
from osgeo import gdal

def vrt_from_dir(input_dir, output_path="./output.vrt", **kwargs):
    inputs = pathlib.Path(input_dir).glob("**/*.tif*")
    vrt_opts = gdal.BuildVRTOptions(**kwargs)
    gdal.BuildVRT(output_path, inputs, options=vrt_opts)


def tiles(input_vrt, output_dir, zoom_levels="10"):
    path = pathlib.Path(input_vrt)
    intermediate = f"{path.parent}/{path.resolve().stem}-byte.vrt"
    subprocess.call(["gdal_translate", "-ot", "Byte", input_vrt, f'{intermediate}'])
    gdal2tiles.generate_tiles(f"{intermediate}", output_dir, zoom=zoom_levels, verbose=True, tile_size=1056)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge to a VRT")
    parser.add_argument("-d", "--input_dir", type=str)
    parser.add_argument("-o", "--output_dir", type=str, default="./")
    parser.add_argument("-n", "--output_name", type=str, default="output.vrt")
    parser.add_argument("-t", "--tile", default=False)
    args = parser.parse_args()

    vrt_path = pathlib.Path(args.output_dir, args.output_name)
    vrt_from_dir(args.input_dir, vrt_path)
    if args.tile:
        tiles(vrt_path, args.output_dir)
