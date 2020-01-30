#!/bin/bash

#python create_raster_preview.py tiles/2005/hkh/LE07_133041_20051112.tif LE07_133041_20051112.jpg


for f in tiles/*.tif
do
    echo "Processing $f file..."
    file=$(echo $f | xargs -l basename | cut -d'.' -f 1)
    if [ ! -f images/${file}.jpg ]; then
        python create_raster_preview.py $f images/${file}.jpg
    else
        echo "Preview exists, skipping."
    fi
    echo ""
done