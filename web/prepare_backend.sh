#!/usr/bin/env bash

# This must be run from the glacier mapping root directory
source .env

# if data or directories don't exist, you'll need to fill them in
# mkdir -p $DATA_DIR/minimal_run/models/
# mkdir -p $DATA_DIR/raw/img_data/mini/
# mkdir -p $DATA_DIR/processed/preds/
# mkdir -p $ROOT_DIR/web/outputs/tiles/
# mkdir -p $ROOT_DIR/web/outputs/tiles/

# data prep for backend
python3 -m web.backend.backend_data -d $DATA_DIR/img_data/ -o $ROOT_DIR/web/frontend/outputs/tiles/ -n output-full.vrt # tiles/ directory must exist
python3 -m web.backend.backend_data -d $DATA_DIR/img_data/ -o $ROOT_DIR/web/frontend/outputs/tiles/ -n output-full.vrt # tiles/ directory must exist
python3 -m web.backend.backend_data -d $DATA_DIR/img_data/ -o $ROOT_DIR/web/frontend/outputs/tiles/ -n output-245.vrt --tile True --bandList 5 4 2
python3 -m web.backend.backend_data -d $DATA_DIR/processed/preds/ -o $ROOT_DIR/web/frontend/outputs/pred_tiles/ -n y_hat.vrt --bandList 1 --tile True # tiles/ directory must exist

# copy tile outputs to $ROOT_DIR/web/outputs/
# you will see results at http://0.0.0.0:4040/web/frontend/index.html
python3 -m web.backend.server & python3 -m frontend_server
