#!/usr/bin/env bash
source /home/kris/glacier_mapping/.env

cd $ROOT_DIR
export split_dir=$DATA_DIR/expers/geographic/splits/
export slice_meta=$DATA_DIR/processed_exper/slices/slices.geojson
export n_folds=9
rm -rf $split_dir
mkdir -p $split_dir

# create train and test geojsons
for i in $( seq 0 $n_folds )
do
    mkdir $split_dir/$i
    python3 -m glacier_mapping.experiment_helpers.geo -s $slice_meta -o $split_dir/$i/
done

# construct different folds
for i in $( seq 0 $n_folds )
do
    python3 -m scripts.geo.conf -i $i -t $ROOT_DIR/conf/geo/postprocess.yaml -o $split_dir/$i/postprocess.yaml
    python3 -m scripts.process_slices -o $split_dir/$i -m $DATA_DIR/processed_exper/slices/slices.geojson -p $split_dir/$i/postprocess.yaml
    cd $split_dir
    tar -zcvf split_$i.tar.gz $i
    cd $ROOT_DIR
done;
