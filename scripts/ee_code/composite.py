
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Get temporal composite by geometry

python3 -m composite --conf le7_2010.yaml
"""""
import ee
import yaml
from datetime import datetime
import geopandas as gpd
import threading
import utils as ut
import json
from addict import Dict


if __name__ == '__main__':
    args = ut.parse_args()
    conf = Dict(yaml.safe_load(open(args.conf, "r")))
    slc_failure_date = datetime.strptime(conf.slc_failure_date, "%Y-%m-%d")
    start_date = datetime.strptime(conf.infer.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(conf.infer.end_date, "%Y-%m-%d")
    ee.Initialize()

    feature_ = json.load(open(conf.infer.geojson, "r"))
    feature = ee.Feature(feature_)

    collection = ee.ImageCollection("LANDSAT/LE07/C01/T1_SR")\
      .map(ut.append_features)\
      .filterDate(start_date, end_date)\
      .filter(feature)\
      .median()

    composite = ee.Algorithms.Landsat.simpleComposite(collection);
    tasks = [ut.export_image(composite, args.conf, description="composite")]
    f_stop = threading.Event()
    ut.display_task_info(tasks, f_stop)
