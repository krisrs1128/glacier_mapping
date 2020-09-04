#! /usr/bin/env python3
# -*- coding: utf-8 -*-
from web.backend.Datasets import load_dataset, get_area_from_geometry
from addict import Dict
from web.backend.log import setup_logging, LOGGER
import web.backend.DataLoader as DL
from web.backend.ServerModelsPytorch import PytorchUNet
import web.backend.Utils as utils
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
DATA_DIR = os.environ["DATA_DIR"]
bottle.TEMPLATE_PATH.insert(0, DATA_DIR + "/views") # let bottle know where we are storing the template files

with open("conf/models.json", "r") as f:
    models = json.load(f)
    model = PytorchUNet(models["benjamins_unet"]["model"], 0)

#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------

def do_options():
    '''This method is necessary for CORS to work (I think --Caleb)
    '''
    bottle.response.status = 204
    return

@app.hook("after_request")
def enable_cors():
    '''From https://gist.github.com/richard-flosi/3789163
    This globally enables Cross-Origin Resource Sharing (CORS) headers for every response from this server.
    '''
    print("after_request called")
    bottle.response.headers['Access-Control-Allow-Origin'] = '*'
    bottle.response.headers['Access-Control-Allow-Methods'] = 'PUT, GET, POST, DELETE, OPTIONS'
    bottle.response.headers['Access-Control-Allow-Headers'] = 'Origin, Accept, Content-Type, X-Requested-With, X-CSRF-Token'

#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------

@app.post("/predPatch")
def pred_patch():
    ''' Method called for POST `/predPatch`'''
    print("pred patch called")
    bottle.response.headers['Content-type'] = 'application/json'
    data = Dict(bottle.request.json)

    # Load the input data sources for the given tile
    name_list = [item["name"] for item in data["classes"]]
    loaded_query = DATASET["data_loader"].get_data_from_extent(data.extent)

    # Run a model on the input data and warp to EPSG:3857
    x, y_hat = model.run(loaded_query["src_img"])
    y_hat, output_bounds = DL.warp_data(
        y_hat.astype(np.float32),
        loaded_query["src_crs"],
        loaded_query["src_transform"],
        loaded_query["src_bounds"]
    )

    # extract geojson associated with the prediction
    y_geo = DL.convert_to_geojson(y_hat[:, :, 0], loaded_query["src_bounds"])
    data["y_geo"] = y_geo

    # Convert images to base64 and return
    data["src_img"] = DL.encode_rgb(x.astype(np.float32))
    data["output_soft"] = DL.encode_rgb((y_hat - y_hat.min()) / y_hat.ptp() * 255)
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



app.route("/predPatch", method="OPTIONS", callback=do_options)
bottle.run(app, host="0.0.0.0", port="8080")
