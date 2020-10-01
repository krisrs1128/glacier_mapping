"""Helpers for Geographic Generalization Experiments

export run_dir=$DATA_DIR/expers/geographic/splits/01/
mkdir -p $run_dir
python3 -m experiment_helpers.geo -d $DATA_DIR/analysis_images/ -o $run_dir
"""
import geopandas as gpd
import pandas as pd
import numpy as np
import shapely
import random
import pathlib
import rasterio
from shapely.ops import cascaded_union
import argparse
import subprocess


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


def grow_region(work_region, init_coords, train_perc=0.8, grow_rate=4000):
    """
    Take a shapely point and buffer it into a circle of fraction train_perc of
    the work region, incrementally growing at rate grow_rate.
    """
    region = cascaded_union(init_coords)

    while True:
        region = region.buffer(grow_rate)
        shared_area = region.intersection(work_region).area / work_region.area
        print(f"shared area: {shared_area}")
        if shared_area > train_perc:
            return region.intersection(work_region)


def geo_split(work_region, proposal_region=None, train_perc=0.8, n_init=2):
    """
    input: - geojson specifying the work area
      - relative size of train / test splits (fraction of working area geojson to assign to train or test)
    output: Two geographically disjoint geojsons, one for train and one for test
    """
    if not proposal_region:
        proposal_region = work_region

    init_coords = []
    for i in range(n_init):
        init_coords.append(random_location(proposal_region))

    train_region = grow_region(work_region, init_coords, train_perc)
    test_region = work_region.difference(train_region)
    return (train_region, test_region)


def create_gdf(polygon, crs=3857):
    return gpd.GeoDataFrame(geometry=[polygon], crs=crs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="create geographic train and test splits")
    parser.add_argument("-s", "--slice_metadata", type=str)
    parser.add_argument("-o", "--output_dir", type=str)
    args = parser.parse_args()

    slice_meta = gpd.read_file(args.slice_metadata)
    work_region = cascaded_union(slice_meta["geometry"])
    work_region = gpd.GeoDataFrame(geometry = [work_region], crs = slice_meta.crs)
    work_region = work_region.to_crs(3857)
    train, test = geo_split(work_region["geometry"][0])

    # convert to geopandas df, and svae to geojson
    train_df = create_gdf(train)
    test_df = create_gdf(test)
    work_region.to_file(pathlib.Path(args.output_dir) / "work_region.geojson", driver="GeoJSON")
    train_df.to_file(pathlib.Path(args.output_dir) / "train.geojson", driver="GeoJSON")
    test_df.to_file(pathlib.Path(args.output_dir) / "test.geojson", driver="GeoJSON")
