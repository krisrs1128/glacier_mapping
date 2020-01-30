#!/bin/bash

mkdir data/
mkdir shapes/
mkdir tiles/
mkdir downloads/

# Setup the local endpoints.js file (this is not in git)
cp endpoints.js endpoints.mine.js

# Copy everything from the web-tool-data blob
cp /mnt/blobfuse/web-tool-data/web_tool/data/* data/
cp /mnt/blobfuse/web-tool-data/web_tool/shapes/* shapes/

# do this to get all data
#cp /mnt/blobfuse/web-tool-data/web_tool/tiles/* web_tool/tiles/

# or something like this to get some data
cp /mnt/blobfuse/web-tool-data/web_tool/tiles/yangon* tiles/


# Unzip what we need
cd tiles/
unzip -q \*.zip
cd ../../
