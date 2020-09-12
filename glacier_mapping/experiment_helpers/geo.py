"""
Helpers for Geographic Generalization Experiments
"""
import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from shapely import Point


def random_location(polygon, N=1):
    points = []
    minx, miny, maxx, maxy = polygon.bounds
    while len(points) < N:
        pnt = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if polygon.contains(pnt):
            points.append(pnt)
    return points


def grow_region(work_region, init_coord, train_perc=0.8, grow_rate=0.1):
    """
    Take a shapely point and buffer it into a circle of fraction train_perc of
    the work region, incrementally growing at rate grow_rate.
    """
    region = init_coord

    while True:
        region = region.buffer(grow_rate)
        shared_area = region.intersection(work_region).area / work_region.area
        if shared_area > train_perc:
            return region


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
