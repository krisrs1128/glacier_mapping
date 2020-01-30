#!/bin/bash

mkdir web_tool/data/
mkdir web_tool/shapes/
mkdir web_tool/tiles/
mkdir web_tool/downloads/

# Setup the local endpoints.js file (this is not in git)
cp web_tool/endpoints.js web_tool/endpoints.mine.js

# Copy everything from the web-tool-data blob
cp /mnt/blobfuse/web-tool-data/web_tool/data/* web_tool/data/
cp /mnt/blobfuse/web-tool-data/web_tool/shapes/* web_tool/shapes/

# do this to get all data
#cp /mnt/blobfuse/web-tool-data/web_tool/tiles/* web_tool/tiles/

# or something like this to get some data
cp /mnt/blobfuse/web-tool-data/web_tool/tiles/yangon* web_tool/tiles/


# Unzip what we need
cd web_tool/tiles/
unzip -q \*.zip
cd ../../
