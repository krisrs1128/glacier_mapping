#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import yaml
from utils import *

if __name__ == '__main__':
    args = parse_args()
    conf = Dict(yaml.safe_load(open(args.conf, "r")))

    gdrive_folder = conf.infer.gdrive_folder
    wrs_path_row = conf.infer.wrs_path_row
    slc_failure_date = datetime.strptime(conf.slc_failure_date, "%Y-%m-%d")
    start_date = datetime.strptime(conf.infer.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(conf.infer.end_date, "%Y-%m-%d")

    #ee.Authenticate()
    ee.Initialize()

    tasks = []

    for (path, row) in wrs_path_row:
        # Get source image for given path row and between given dates
        collection = ee.ImageCollection('LANDSAT/LE07/C01/T1_RT').filterDate(start_date, end_date).filter(ee.Filter.eq('WRS_ROW', int(row))).filter(ee.Filter.eq('WRS_PATH', int(path)))
        source = ee.Image(collection.sort('CLOUD_COVER').first())

        source_date_acquired = source.getInfo()['properties']['DATE_ACQUIRED']
        source_date_acquired = datetime.strptime(source_date_acquired, '%Y-%m-%d')

        # Check if the image is affected by SLC failure and correct them
        if source_date_acquired >= slc_failure_date:
            fill = get_fill_image(source)
            img = gapfill(source, fill)
        else:
            img = source

        # Generate additional features
        img = add_index(img, ["B4", "B3"], "ndvi")
        img = add_index(img, ["B2", "B5"], "ndsi")
        img = add_index(img, ["B4", "B5"], "ndwi")
        # Add slope and elevation
        elevation = ee.Image('CGIAR/SRTM90_V4').select('elevation')
        slope = ee.Terrain.slope(elevation);
        img = ee.Image.cat([img, elevation, slope]);

        # Export image
        img = img.toFloat();
        task = export_image(img, folder = gdrive_folder)
        tasks.append(task)

    f_stop = threading.Event()
    display_task_info(tasks, f_stop)
