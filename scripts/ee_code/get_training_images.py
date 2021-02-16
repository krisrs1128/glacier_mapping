#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import yaml
from utils import *
from addict import Dict

if __name__ == '__main__':
    args = parse_args()
    conf = Dict(yaml.safe_load(open(args.conf, "r")))
    slc_failure_date = datetime.strptime(conf.slc_failure_date, "%Y-%m-%d")
    ee.Initialize()

    tasks = []
    for image_id in conf.train.image_ids:
        # Get landsat 7 images
        source = ee.Image('LANDSAT/LE07/C01/T1_RT/' + image_id)
        tasks += [fetch_task(source, slc_failure_date, conf.train.gdrive_folder)]

    f_stop = threading.Event()
    display_task_info(tasks, f_stop)
