"""
Helpers for Geographic Generalization Experiments

export run_dir=$DATA_DIR/expers/geographic/splits/01/
mkdir -p $run_dir
mkdir -p $run_dir
#python3 -m experiment_helpers.geo -d $DATA_DIR/analysis_images/ -o $run_dir
python3 -m experiment_helpers.geo -d $DATA_DIR/expers/geographic/test_input/ -o $run_dir
"""
import geopandas as gpd
import pandas as pd
import numpy as np
import shapely
import random
import shutil
import pathlib
import rasterio
from shapely.ops import cascaded_union
import argparse
import subprocess


def extract_work_region(tiles_dir):
    """
    input: directory containing tiles
    output: geojson of the cascaded union of bounding boxes for the tiles in
    the directory
    """
    tile_paths = list(pathlib.Path(tiles_dir).glob("*.tif*"))
    bboxes = []
    for path in tile_paths:
        imgf = rasterio.open(path)
        bbox = shapely.geometry.box(*imgf.bounds)
        bboxes.append(bbox)
    return cascaded_union(bboxes)


def random_location(polygon):
    """
    Get a random shapely.Point within a polygon
    """
    points = []
    minx, miny, maxx, maxy = polygon.bounds

    while True:
        ux, uy = random.uniform(minx, maxx), random.uniform(miny, maxy)
        pnt = shapely.geometry.Point(ux, uy)
        if polygon.contains(pnt):
            return pnt


def grow_region(work_region, init_coord, train_perc=0.8, grow_rate=1000):
    """
    Take a shapely point and buffer it into a circle of fraction train_perc of
    the work region, incrementally growing at rate grow_rate.
    """
    region = init_coord

    while True:
        region = region.buffer(grow_rate)
        shared_area = region.intersection(work_region).area / work_region.area
        print(f"shared area: {shared_area}")
        if shared_area > train_perc:
            return region.intersection(work_region)


def geo_split(work_region, train_perc=0.8):
    """
    input: - geojson specifying the work area
      - relative size of train / test splits (fraction of working area geojson to assign to train or test)
    output: Two geographically disjoint geojsons, one for train and one for test
    """
    init_coord = random_location(work_region)
    train_region = grow_region(work_region, init_coord, train_perc)
    test_region = work_region.difference(train_region)
    return (train_region, test_region)


def reproject_directory(input_dir, output_dir, dst_epsg=3857):
    """
    Warp all Tiffs from one directory to 3857
    """
    inputs = pathlib.Path(input_dir).glob("*.tif*")
    for im_path in inputs:
        print(f"reprojecting {str(im_path)}")
        loaded_im = rasterio.open(im_path)
        output_path = pathlib.Path(output_dir, f"{im_path.stem}-warped.tif")
        subprocess.call(["gdalwarp", "-s_srs", str(loaded_im.crs), "-t_srs",
                         f"EPSG:{dst_epsg}", str(im_path),
                         "-wo", "NUM_THREADS=ALL_CPUS", str(output_path)])

def create_gdf(polygon, crs=3857):
    return gpd.GeoDataFrame(geometry=[polygon], crs=crs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="create geographic train and test splits")
    parser.add_argument("-d", "--input_dir", type=str)
    parser.add_argument("-o", "--output_dir", type=str)
    parser.add_argument("-r", "--reproject", type=bool, default=False)
    args = parser.parse_args()

    # reproject, if requested
    if args.reproject:
        reproject_directory(args.input_dir, args.output_dir, 3857)
    else:
        for f in pathlib.Path(args.input_dir).glob("*.tif*"):
            shutil.copy(f, pathlib.Path(args.output_dir))

    work_region = extract_work_region(args.output_dir)
    train, test = geo_split(work_region)

    # convert to geopandas df, and svae to geojson
    work_df = create_gdf(work_region)
    train_df = create_gdf(train)
    test_df = create_gdf(test)
    work_df.to_file(pathlib.Path(args.output_dir) / "work_region.geojson", driver="GeoJSON")
    train_df.to_file(pathlib.Path(args.output_dir) / "train.geojson", driver="GeoJSON")
    test_df.to_file(pathlib.Path(args.output_dir) / "test.geojson", driver="GeoJSON")
