#!/usr/bin/env bash

# This must be run from the glacier mapping root directory
source .env

# data prep for backend
python3 -m web_backend.backend_data -d $DATA_DIR/raw/img_data/mini/ -o $ROOT_DIR/web_tool/outputs/tiles/ -n output-full.vrt # tiles/ directory must exist
python3 -m web_backend.backend_data -d $DATA_DIR/raw/img_data/mini/ -o $ROOT_DIR/web_tool/outputs/tiles/ -n output-245.vrt --tile True --bandList 5 4 2
python3 -m web.backend.backend_data -d $DATA_DIR/processed/preds/ -o $ROOT_DIR/web_tool/outputs/pred_tiles/ -n y_hat.vrt --bandList 1 --tile True # tiles/ directory must exist

# copy tile outputs to $ROOT_DIR/web_tool/outputs/
python3 -m web.backend.server & python3 -m frontend_server
