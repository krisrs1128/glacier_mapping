#!/usr/bin/env bash

# This must be run from the glacier mapping root directory
source .env

# if data or directories don't exist, you'll need to fill them in
mkdir -p $DATA_DIR/web/basemap/

# data prep for backend
python3 -m web.backend.backend_data -d $DATA_DIR/unique_tiles/ -o $DATA_DIR/web/basemap/ -n output-no-elevation.vrt --reproject True --bandList 1 2 3 4 5 6 7 8 9 10 11 12 13
python3 -m web.backend.backend_data -d $DATA_DIR/unique_tiles/ -o $DATA_DIR/web/basemap/ -n output-elevation.vrt --reproject True --bandList 14 15

python3 -m web.backend.backend_data -d $DATA_DIR/unique_tiles/warped/ -o $DATA_DIR/web/basemap/ -n output-no-elevation.vrt --bandList 1 2 3 4 5 6 7 8 9 10 11 12 13
python3 -m web.backend.backend_data -d $DATA_DIR/unique_tiles/warped/ -o $DATA_DIR/web/basemap/ -n output-11.vrt --bandList 11
python3 -m web.backend.backend_data -d $DATA_DIR/unique_tiles/warped/ -o $DATA_DIR/web/basemap/ -n output-elevation.vrt --bandList 14 15

python3 -m web.backend.backend_data -d $DATA_DIR/unique_tiles/warped/ -o $DATA_DIR/web/basemap/ -n output-245.vrt --bandList 5 4 2

## command to combine output-no-elevation and output-elevation
gdal_merge.py -of VRT -separate -o output-full.vrt output-elevation.vrt output-no-elevation.vrt

cd $DATA_DIR/web/basemap/
gdal_translate -ot Byte output-245.vrt output-245-byte.vrt

for i in $( seq 8 16 )
do
    gdal2tiles.py -z $i --processes 10 output-245-byte.vrt .
    mv $i $ROOT_DIR/web/basemap/
done;
