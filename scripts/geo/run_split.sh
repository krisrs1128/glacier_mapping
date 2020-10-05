#!/bin/bash
tar -zxvf split_${1}.tar.gz

git clone https://github.com/krisrs1128/glacier_mapping.git
cd glacier_mapping
source .env

cd scripts
python3 train.py -d ../../${1}/ -c ../conf/train.yaml  -p ../../${1}/postprocess.yaml -r geo
cd ../../
tar -zcvf runs_${1}.tar.gz ${1}/runs/
