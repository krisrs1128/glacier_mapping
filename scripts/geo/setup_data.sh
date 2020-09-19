#!/usr/bin/env bash
echo $1

if [$1 == "test"]
then
    export input_dir=$DATA_DIR/analysis_images
else
    export input_dir=$DATA_DIR/expers/geographic/test_input/
fi

cd $ROOT_DIR
export split_dir=$DATA_DIR/expers/geographic/splits/
rm -rf $split_dir
mkdir -p $split_dir

# create train and test geojsons
mkdir $split_dir/1/
python3 -m glacier_mapping.experiment_helpers.geo -d $input_dir -o $split_dir/1/ -r True

for i in {2..10}; do
    echo $i
    mkdir $split_dir/$i
    python3 -m glacier_mapping.experiment_helpers.geo -d $split_dir/1/ -o $split_dir/$i/
done

# slice
python3 -m scripts.make_slices -m conf/geo/mask.yaml -o $DATA_DIR/expers/geographic/

# construct different folds
for i in {1..10}; do
    python3 -m scripts.geo.conf -i $i -t conf/geo/postprocess.yaml -o $split_dir/$i/postprocess.yaml
    python3 -m scripts.process_slices -o $split_dir/$i -m $DATA_DIR/expers/geographic/slices/slices.geojson -p $split_dir/$i/postprocess.yaml
done;
