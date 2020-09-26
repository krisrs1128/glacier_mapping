#!/usr/bin/env bash
source /home/kris/glacier_mapping/.env

cd $ROOT_DIR
export split_dir=$DATA_DIR/expers/geographic/splits/
export slice_meta=$DATA_DIR/processed/slices/slices.geojson
rm -rf $split_dir
mkdir -p $split_dir

# create train and test geojsons
for i in $( seq 1 $n_folds)
do
    echo $i
    mkdir $split_dir/$i
    python3 -m glacier_mapping.experiment_helpers.geo -s $slice_meta -o $split_dir/$i/
done

# construct different folds
for i in $( seq 1 $n_folds)
do
    python3 -m scripts.geo.conf -i $i -t conf/geo/postprocess.yaml -o $split_dir/$i/postprocess.yaml
    python3 -m scripts.process_slices -o $split_dir/$i -m $DATA_DIR/processed/slices/slices.geojson -p $split_dir/$i/postprocess.yaml
done;
