#!/usr/bin/env bash
cd $ROOT_DIR
export split_dir=$DATA_DIR/expers/geographic/splits/
mkdir -p $split_dir

# create train and test geojsons
mkdir $split_dir/1/
python3 -m glacier_mapping.experiment_helpers.geo -d $DATA_DIR/expers/geographic/test_input/ -o $split_dir/1/ -r True

for i in 2 .. 10; do
    echo $i
    mkdir $split_dir/$i
    python3 -m glacier_mapping.experiment_helpers.geo -d $split_dir/1/ -o $split_dir/$i/
done

# slice
python3 -m scripts.make_slices -m conf/geo/mask.yaml -o $DATA_DIR/expers/geographic/

# construct different folds
for i in 1 .. 10; do
    python3 -m scripts.geo.conf -i $i -t conf/geo/postprocess.yaml -o $split_dir/postprocess.yaml
    python3 -m scripts.process_slices -o $split_dir/$i -p conf/geo/process_$i.yaml
done;
