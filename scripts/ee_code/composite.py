#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Get temporal composite by geometry

Example Usage
python3 -m composite --conf khumbu_2010.yaml

where khumbu_2010.yaml has the form

slc_failure_date: '2003-05-31'
gdrive_folder: '2010_09_12'
geojson: 'khumbu.geojson'
start_date: '2010-09-01'
end_date: '2010-12-31'

and 'khumbu.geojson' is a geojson file outlining the khumbu basin.
"""
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
    start_date = datetime.strptime(conf.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(conf.end_date, "%Y-%m-%d")
    ee.Initialize()

    feature_ = json.load(open(conf.geojson, "r"))
    feature = ee.Geometry(feature_["features"][0]["geometry"])

    collection = ee.ImageCollection('LANDSAT/LE07/C01/T1_SR')\
      .filterBounds(feature)\
      .map(ut.append_features)\
      .filterDate(start_date, end_date)

    composite = collection.median()\
      .clip(feature)

    tasks = [ut.export_image(composite, conf.gdrive_folder, description="composite")]
    f_stop = threading.Event()
    ut.display_task_info(tasks, f_stop)
