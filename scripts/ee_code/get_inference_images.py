#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Get source image for given path row and between given dates

Fetches image with lowest cloud cover between two time points

Example Usage
python3 -m get_inference_images --conf gdrive.yaml
"""
from addict import Dict
from datetime import datetime
import ee
import threading
import utils as ut
import yaml

if __name__ == '__main__':
    args = ut.parse_args()
    conf = Dict(yaml.safe_load(open(args.conf, "r")))
    slc_failure_date = datetime.strptime(conf.slc_failure_date, "%Y-%m-%d")
    start_date = datetime.strptime(conf.infer.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(conf.infer.end_date, "%Y-%m-%d")
    ee.Initialize()

    tasks = []
    for (path, row) in conf.infer.wrs_path_row:
        collection = ee.ImageCollection('LANDSAT/LE07/C01/T1_SR')\
          .filterDate(start_date, end_date)\
          .filter(ee.Filter.eq('WRS_ROW', int(row)))\
          .filter(ee.Filter.eq('WRS_PATH', int(path)))
        source = ee.Image(collection.sort('CLOUD_COVER').first())
        tasks += [ut.fetch_task(source, slc_failure_date, conf.train.gdrive_folder)]

    f_stop = threading.Event()
    ut.display_task_info(tasks, f_stop)
