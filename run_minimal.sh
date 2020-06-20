#!/usr/bin/env bash

# All options below are default, and technically don't need to be specified.
source .env
python3 -m src.data.mask -m conf/masking_paths.yaml
python3 -m src.data.slice -m $DATA_DIR/processed/masks/mask_metadata.csv -o $DATA_DIR/processed/slices/
python3 -m src.data.process_slices -c conf/postprocess.yaml -d $DATA_DIR/data/processed/slices/ -m $DATA_DIR/processed/slices/slices_0-100.geojson -o $DATA_DIR/processed
python3 -m src.train -n minimal_run -c conf/train.yaml
python3 -m src.infer -m data/runs/minimal_run/models/model_5.pt -i data/raw/img_data/2010/nepal/Nepal_139041_20111225.tif
python3 -m web_backend.backend_data -d /mnt/blobfuse/glaciers/raw/img_data/mini/ -o /mnt/blobfuse/glaciers/processed/tiles/ -n output-full.vrt # tiles/ directory must exist
python3 -m web_backend.backend_data -d /mnt/blobfuse/glaciers/raw/img_data/mini/ -o /mnt/blobfuse/glaciers/processed/tiles/ -n output-245.vrt --tile True --bandList 5 4 2
