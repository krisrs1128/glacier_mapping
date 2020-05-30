#!/usr/bin/env bash

# All options below are default, and technically don't need to be specified.
source .env
python3 -m src.data.mask -m conf/masking_paths.yaml
python3 -m src.data.slice -m data/processed/masks/mask_metadata.csv -o data/processed/slices/
python3 -m src.data.process_slices -c conf/postprocess.yaml -d data/processed/slices/ -m data/processed/slices/slices_0-100.geojson -o data/processed
python3 -m src.train -n minimal_run -c conf/train.yaml
python3 -m src.infer -m data/runs/minimal_run/models/model_5.pt -i data/raw/img_data/2010/nepal/Nepal_139041_20111225.tif
