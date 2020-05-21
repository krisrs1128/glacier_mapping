#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# pylint: disable=E1137,E1136,E0110,E1101
from web_backend.Datasets import load_dataset, get_area_from_geometry
from web_backend.Session import Session, manage_session_folders, SESSION_FOLDER
from addict import Dict
from web_backend.log import setup_logging, LOGGER
import web_backend.DataLoader as DL
from web_backend.ServerModelsPytorch import PytorchUNet
import web_backend.Utils as utils
import argparse
import beaker.middleware
import bottle
import cheroot.wsgi
import cv2
import fiona
import fiona.transform
import joblib
import json
import numpy as np
import os
import rasterio
import rasterio.warp
import sys
app = bottle.Bottle()

DATASET = load_dataset()
REPO_DIR = os.environ["REPO_DIR"]
bottle.TEMPLATE_PATH.insert(0, REPO_DIR + "/views") # let bottle know where we are storing the template files

with open("conf/models.json", "r") as f:
    models = json.load(f)
    model = PytorchUNet(models["benjamins_unet"]["model"], 0)

#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------

@app.route('/', method = 'OPTIONS')
@app.route('/<path:path>', method = 'OPTIONS')
def options_handler(path = None):
    return

@app.hook("after_request")
def enable_cors():
    '''From https://gist.github.com/richard-flosi/3789163

    This globally enables Cross-Origin Resource Sharing (CORS) headers for every response from this server.
    '''
    bottle.response.headers['Access-Control-Allow-Origin'] = '*'
    bottle.response.headers['Access-Control-Allow-Methods'] = 'PUT, GET, POST, DELETE, OPTIONS'
    bottle.response.headers['Access-Control-Allow-Headers'] = 'Origin, Accept, Content-Type, X-Requested-With, X-CSRF-Token'

#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------

def reset_model():
    bottle.response.content_type = 'application/json'
    data = bottle.request.json
    data["remote_address"] = bottle.request.client_ip

    initial_reset = data.get("initialReset", False)
    if not initial_reset:
        SESSION_HANDLER.get_session(bottle.request.session.id).add_entry(data) # record this interaction
        SESSION_HANDLER.get_session(bottle.request.session.id).save(data["experiment"])

    SESSION_HANDLER.get_session(bottle.request.session.id).reset()

    data["message"] = "Reset model"
    data["success"] = True

    bottle.response.status = 200
    return json.dumps(data)


def retrain_model():
    bottle.response.content_type = 'application/json'
    data = bottle.request.json
    data["remote_address"] = bottle.request.client_ip

    success, message = SESSION_HANDLER.get_session(bottle.request.session.id).model.retrain(**data["retrainArgs"])

    if success:
        bottle.response.status = 200
        encoded_model_fn = SESSION_HANDLER.get_session(bottle.request.session.id).save(data["experiment"])
        data["cached_model"] = encoded_model_fn
        SESSION_HANDLER.get_session(bottle.request.session.id).add_entry(data) # record this interaction
    else:
        data["error"] = message
        bottle.response.status = 500

    data["message"] = message
    data["success"] = success

    return json.dumps(data)


@app.post("/predPatch")
def pred_patch():
    ''' Method called for POST `/predPatch`'''
    bottle.response.content_type = 'application/json'
    data = Dict(bottle.request.json)

    # Load the input data sources for the given tile
    extent = data.extent
    dataset = data.dataset
    name_list = [item["name"] for item in data["classes"]]
    loaded_query = DATASET["data_loader"].get_data_from_extent(extent)

    # Run a model on the input data adn warp to EPSG:3857
    output = model.run(loaded_query["src_img"])
    y_hat, output_bounds = DL.warp_data(
        output["y"].astype(np.float32),
        loaded_query["src_crs"],
        loaded_query["src_transform"],
        loaded_query["src_bounds"]
    )

    # ------------------------------------------------------
    # Step 5
    #   Convert images to base64 and return
    # ------------------------------------------------------
    img_soft = np.round(utils.class_prediction_to_img(y_hat))
    data["src_img"] = DL.encode_rgb(np.float32(output["x"]))
    data["output_soft"] = DL.encode_rgb(img_soft)
    bottle.response.status = 200
    return json.dumps(data)


@app.post("/predTile")
def pred_tile():
    ''' Method called for POST `/predTile`'''
    bottle.response.content_type = 'application/json'
    data = bottle.request.json
    data["remote_address"] = bottle.request.client_ip

    SESSION_HANDLER.get_session(bottle.request.session.id).add_entry(data) # record this interaction

    # Inputs
    geom = data["polygon"]
    class_list = data["classes"]
    name_list = [item["name"] for item in class_list]
    dataset = data["dataset"]
    zone_layer_name = data["zoneLayerName"]

    if dataset not in DATASET:
        raise ValueError("Dataset doesn't seem to be valid, do the dataset in js/tile_layers.js correspond to those in TileLayers.py")

    try:
        naip_data, raster_profile, raster_transform, raster_bounds, raster_crs = DATASET[dataset]["data_loader"].get_data_from_shape(geom["geometry"])
        naip_data = np.rollaxis(naip_data, 0, 3)
        shape_area = get_area_from_geometry(geom["geometry"])
    except NotImplementedError as e:
        bottle.response.status = 400
        return json.dumps({"error": "Cannot currently download imagery with 'Basemap' based dataset"})

    output = SESSION_HANDLER.get_session(bottle.request.session.id).model.run(naip_data, geom, True)
    output_hard = output.argmax(axis=2)
    print("Finished, output dimensions:", output.shape)

    # apply nodata mask from naip_data
    nodata_mask = np.sum(naip_data == 0, axis=2) == naip_data.shape[2]
    output_hard[nodata_mask] = 255
    vals, counts = np.unique(output_hard[~nodata_mask], return_counts=True)

    # ------------------------------------------------------
    # Step 4
    #   Convert images to base64 and return
    # ------------------------------------------------------
    tmp_id = utils.get_random_string(8)
    img_hard = np.round(utils.class_prediction_to_img(output * 255,0)).astype(np.uint8)
    img_hard = cv2.cvtColor(img_hard, cv2.COLOR_RGB2BGRA)
    img_hard[nodata_mask] = [0,0,0,0]

    img_hard, img_hard_bounds = DL.warp_data_to_3857(img_hard, raster_crs, raster_transform, raster_bounds, resolution=10)

    cv2.imwrite(os.path.join(REPO_DIR, "downloads/%s.png" % (tmp_id)), img_hard)
    data["downloadPNG"] = "downloads/%s.png" % (tmp_id)

    new_profile = raster_profile.copy()
    new_profile['driver'] = 'GTiff'
    new_profile['dtype'] = 'uint8'
    new_profile['compress'] = "lzw"
    new_profile['count'] = 1
    new_profile['transform'] = raster_transform
    new_profile['height'] = naip_data.shape[0]
    new_profile['width'] = naip_data.shape[1]
    new_profile['nodata'] = 255
    f = rasterio.open(os.path.join(REPO_DIR, "downloads/%s.tif" % (tmp_id)), 'w', **new_profile)
    f.write(output_hard.astype(np.uint8), 1)
    f.close()
    data["downloadTIFF"] = "downloads/%s.tif" % (tmp_id)

    f = open(os.path.join(REPO_DIR, "downloads/%s.txt" % (tmp_id)), "w")
    f.write("Class id\tClass name\tPercent area\tArea (km^2)\n")
    for i in range(len(vals)):
        pct_area = (counts[i] / np.sum(counts))
        if shape_area is not None:
            real_area = shape_area * pct_area
        else:
            real_area = -1
        f.write("%d\t%s\t%0.4f%%\t%0.4f\n" % (vals[i], name_list[vals[i]], pct_area*100, real_area))
    f.close()
    data["downloadStatistics"] = "downloads/%s.txt" % (tmp_id)

    bottle.response.status = 200
    return json.dumps(data)


@app.post("/getInput")
def get_input():
    ''' Method called for POST `/getInput`
    '''
    bottle.response.content_type = 'application/json'
    data = Dict(bottle.request.json)
    data["remote_address"] = bottle.request.client_ip

    SESSION_HANDLER.get_session(bottle.request.session.id).add_entry(data) # record this interaction

    # Inputs
    extent = data.extent
    data_id = data.dataset.metadata.id

    if data_id not in DATASET:
        raise ValueError("Dataset doesn't seem to be valid, please check Dataset.py")

    loaded_query = DATASET[data_id]["data_loader"].get_data_from_extent(extent)
    img_data, img_bounds = DL.warp_data_to_3857(**loaded_query)
    data["input_rgb"] = DL.encode_rgb(img_data[:, :, [6, 3, 1]])
    bottle.response.status = 200
    return json.dumps(data)


bottle.run(app, host="localhost", port="4446")
