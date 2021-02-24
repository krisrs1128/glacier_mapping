"""
Get source image for given path row and between given dates

Fetches image with lowest cloud cover between two time points

Example Usage
python3 -m get_inference_images --conf gdrive.yaml
"""
import json
from datetime import datetime
from addict import Dict
import yaml
import ee
import utils as ut

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

    clearest = collection.sort("CLOUD_COVER").first()\
        .clip(feature)

    tasks = [ut.export_image(clearest, conf.gdrive_folder, conf.filename)]
