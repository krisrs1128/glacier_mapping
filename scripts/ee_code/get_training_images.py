#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Get landsat 7 images by ID

Example Usage:
python3 -m get_training_images --conf gdrive.yaml
"""
from addict import Dict
from datetime import datetime
import ee
import threading
import utils import as ut
import yaml

if __name__ == '__main__':
    args = ut.parse_args()
    conf = Dict(yaml.safe_load(open(args.conf, "r")))
    slc_failure_date = datetime.strptime(conf.slc_failure_date, "%Y-%m-%d")
    ee.Initialize()

    tasks = []
    for image_id in conf.train.image_ids:
        source = ee.Image('LANDSAT/LE07/C01/T1_RT/' + image_id)
        tasks += [ut.fetch_task(source, slc_failure_date, conf.train.gdrive_folder)]

    f_stop = threading.Event()
    ut.display_task_info(tasks, f_stop)
