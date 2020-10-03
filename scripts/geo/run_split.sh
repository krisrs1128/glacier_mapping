#!/bin/bash
tar -zxvf splits.tar.gz
pip3 install git+https://github.com/krisrs1128/glacier_mapping.git

git clone https://github.com/krisrs1128/glacier_mapping.git
cd glacier_mapping
source .env

cd scripts
ix=$((1+ ${1}))
python3 train.py -d ../../splits/$ix/ -c ../conf/train.yaml  -p ../../splits/$ix/postprocess.yaml -r geo
