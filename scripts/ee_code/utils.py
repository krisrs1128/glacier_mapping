#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import ee
import argparse
import json
from addict import Dict
from collections import Counter
from datetime import datetime
import threading

def export_image(ee_image, folder, crs = 'EPSG:32644'):
    task = ee.batch.Export.image.toDrive(image = ee_image.float(),  
                                     region = ee_image.geometry(),  
                                     description = ee_image.getInfo()['properties']['system:index'],
                                     folder = folder,
                                     scale = 30,
                                     crs = crs)
    task.start()
    return task

def add_index(ee_image, bands, band_name):
    new_band = ee_image.normalizedDifference(bands).rename(band_name)
    return ee_image.addBands(new_band)

def get_fill_image(ee_image):
    source_wrs_row = ee_image.get("WRS_ROW")
    source_wrs_path = ee_image.get("WRS_PATH")
    fill = ee.ImageCollection('LANDSAT/LE07/C01/T1_RT').filterDate("2000-01-01", "2000-12-31").filter(ee.Filter.eq('WRS_ROW', source_wrs_row)).filter(ee.Filter.eq('WRS_PATH', source_wrs_path))
    fill_img = ee.Image(fill.sort('CLOUD_COVER').first())
    return fill_img

def gapfill(source, fill, kernel_size = 10, upscale = True):
    min_scale = 1/3;
    max_scale = 3;
    min_neighbours = 64;
    # Apply the USGS L7 Phase-2 Gap filling protocol, using a single kernel size.
    kernel = ee.Kernel.square(kernel_size * 30, "meters", False)
    # Find the pixels common to both scenes.
    common = source.mask().And(fill.mask())
    fc = fill.updateMask(common)
    sc = source.updateMask(common)
    # Find the primary scaling factors with a regression.
    # Interleave the bands for the regression.  This assumes the bands have the same names.
    regress = fc.addBands(sc)
    regress = regress.select(regress.bandNames().sort())
    ratio = 5
    if upscale:
        fit = regress.reduceResolution(ee.Reducer.median(), False, 500).reproject(regress.select(0).projection().scale(ratio, ratio)).reduceNeighborhood(ee.Reducer.linearFit().forEach(source.bandNames()), kernel, 'kernel', False).unmask().reproject(regress.select(0).projection().scale(ratio, ratio))
    else:
        fit = regress.reduceNeighborhood(ee.Reducer.linearFit().forEach(source.bandNames()), kernel, 'kernel', False)
    offset = fit.select(".*_offset")
    scale = fit.select(".*_scale")
    # Find the secondary scaling factors using just means and stddev
    reducer = ee.Reducer.mean().combine(ee.Reducer.stdDev(), "", True)
    if upscale:
        src_stats = source.reduceResolution(ee.Reducer.median(), False, 500).reproject(regress.select(0).projection().scale(ratio, ratio)).reduceNeighborhood(reducer, kernel, 'kernel', False).reproject(regress.select(0).projection().scale(ratio, ratio))
        fill_stats = fill.reduceResolution(ee.Reducer.median(), False, 500).reproject(regress.select(0).projection().scale(ratio, ratio)).reduceNeighborhood(reducer, kernel, 'kernel', False).reproject(regress.select(0).projection().scale(ratio, ratio))
    else:
        src_stats = source.reduceNeighborhood(reducer, kernel, 'kernel', False)
        fill_stats = fill.reduceNeighborhood(reducer, kernel, 'kernel', False)
    scale2 = src_stats.select(".*stdDev").divide(fill_stats.select(".*stdDev"))
    offset2 = src_stats.select(".*mean").subtract(fill_stats.select(".*mean").multiply(scale2))
    invalid = scale.lt(min_scale).Or(scale.gt(max_scale))
    scale = scale.where(invalid, scale2)
    offset = offset.where(invalid, offset2)
    # When all else fails, just use the difference of means as an offset.
    invalid2 = scale.lt(min_scale).Or(scale.gt(max_scale))
    scale = scale.where(invalid2, 1)
    offset = offset.where(invalid2, src_stats.select(".*mean").subtract(fill_stats.select(".*mean")))
    # Apply the scaling and mask off pixels that didn't have enough neighbors.
    count = common.reduceNeighborhood(ee.Reducer.count(), kernel, 'kernel', True)
    scaled = fill.multiply(scale).add(offset).updateMask(count.gte(min_neighbours))
    return source.unmask(scaled, True)

def get_image_info(ee_image):
    print(ee_image)
    print(json.dumps(ee_image.getInfo(), indent=2))

def display_task_info(tasks, f_stop):
    task_info = []
    for task in tasks:
        task_info.append(task.status()['state'])
    if (len(set(task_info)) == 1) and ( (task_info[0] == "COMPLETED") or 
                                        (task_info[0] == "CANCEL_REQUESTED") or
                                        (task_info[0] == "CANCELLED") ):
        f_stop.set()
        f_stop.clear()
        print("All tasks completed succesfully")
    print(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    for key, val in dict(Counter(task_info)).items():
        print(f"\t{key}: {val} out of {len(task_info)}")
    if not f_stop.is_set():
        threading.Timer(60, display_task_info, [tasks, f_stop]).start()

def parse_args():
    parser = argparse.ArgumentParser(description="Download satellite images from google earth engine using python API")
    parser.add_argument("-c", "--conf", type=str)
    args = parser.parse_args()
    return args
