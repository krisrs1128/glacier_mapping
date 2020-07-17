#!/usr/bin/env bash

# All options below are default, and technically don't need to be specified.
source .env
python3 -m src.data.mask -m conf/masking_paths.yaml
python3 -m src.data.slice -m $DATA_DIR/processed/masks/mask_metadata.csv -o $DATA_DIR/processed/slices/
python3 -m src.data.process_slices -c conf/postprocess.yaml -d $DATA_DIR/processed/slices/ -m $DATA_DIR/processed/slices/slices_0-100.geojson -o $DATA_DIR/processed
python3 -m src.train -n minimal_run -c conf/train.yaml
python3 -m src.infer -m $DATA_DIR/runs/minimal_run/models/model_305.pt -i $DATA_DIR/raw/img_data/mini/LE07_143039_20051001.tif -o  $DATA_DIR/processed/preds/
python3 -m web_backend.backend_data -d $DATA_DIR/raw/img_data/mini/ -o $ROOT_DIR/web_tool/outputs/tiles/ -n output-full.vrt # tiles/ directory must exist
python3 -m web_backend.backend_data -d $DATA_DIR/raw/img_data/mini/ -o $ROOT_DIR/web_tool/outputs/tiles/ -n output-245.vrt --tile True --bandList 5 4 2

# mv the output of infer to the pred_tiles folder
python3 -m web_backend.backend_data -d $DATA_DIR/processed/preds/ -o $ROOT_DIR/web_tool/outputs/pred_tiles/ -n y_hat.vrt --bandList 1 --tile True # tiles/ directory must exist

# copy tile outputs to $ROOT_DIR/web_tool/outputs/
python3 -m web_backend.server & python3 -m frontend_server
