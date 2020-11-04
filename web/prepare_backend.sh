#!/usr/bin/env bash

# This must be run from the glacier mapping root directory
source .env

# if data or directories don't exist, you'll need to fill them in
mkdir -p $DATA_DIR/web/basemap/

# data prep for backend
python3 -m web.backend.backend_data -d $DATA_DIR/unique_tiles/ -o $DATA_DIR/web/basemap/ -n output-full.vrt --reproject True
python3 -m web.backend.backend_data -d $DATA_DIR/unique_tiles/warped/ -o $DATA_DIR/web/basemap/ -n output-245.vrt --bandList 5 4 2

python3 -m web.backend.backend_data -d $DATA_DIR/web/tests/ -o $DATA_DIR/web/tests/ -n output-full.vrt --reproject True;
python3 -m web.backend.backend_data -d $DATA_DIR/web/tests/warped/ -o $DATA_DIR/web/tests/ -n output-245.vrt --bandList 5 4 2

cd $DATA_DIR/web/basemap/
gdal_translate -ot Byte output-245.vrt output-245-byte.vrt

for i in $( seq 8 16 )
do
    gdal2tiles.py -z $i --processes 10 output-245-byte.vrt .
    cp -r $i $ROOT_DIR/web/basemap/
done;

# you will see results at http://0.0.0.0:4040/web/frontend/index.html
python3 -m web.frontend_server & python3 -m web.backend.server
