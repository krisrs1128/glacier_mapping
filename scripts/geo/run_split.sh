#!/usr/bin/env bash
tar -zxvf splits.tar.gz

git clone https://github.com/krisrs1128/glacier_mapping.git
cd glacier_mapping
source .env

cd scripts
python3 train.py -d /home/ksankaran/splits/1/ -c ../conf/train.yaml  -p /home/ksankaran/splits/1/postprocess.yaml -r geo
