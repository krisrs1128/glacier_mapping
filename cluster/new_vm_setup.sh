#!/bin/bash

# update blobfuse version
sudo apt-get update
sudo apt-get install blobfuse

# install our python geospatial library dependencies
source activate /data/anaconda/envs/py36
conda install rasterio fiona shapely rtree
pip install --user --upgrade bottle einops mercantile rasterio
pip install --user pandas geopandas Pillow addict pyyaml tqdm seaborn utm beaker cheroot rpyc
source deactivate

# sometimes the GPU doesn't work when we first provision the VM
sudo shutdown -r now