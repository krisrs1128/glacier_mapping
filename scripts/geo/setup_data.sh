#!/usr/bin/env bash
echo $1
source /home/kris/glacier_mapping/.env

if [ $1 == "test" ];
then
    export input_dir=$DATA_DIR/expers/geographic/test_input/
    export n_folds=3
    export mask_path=conf/geo/mask_small.yaml
else
    export input_dir=$DATA_DIR/analysis_images
    export n_folds=10
    export mask_path=conf/geo/mask.yaml
fi

cd $ROOT_DIR
export split_dir=$DATA_DIR/expers/geographic/splits/
rm -rf $split_dir
mkdir -p $split_dir

# create train and test geojsons
mkdir $split_dir/1/
python3 -m glacier_mapping.experiment_helpers.geo -d $input_dir -o $split_dir/1/ -r True

for i in $( seq 2 $n_folds)
    echo $i
    mkdir $split_dir/$i
    python3 -m glacier_mapping.experiment_helpers.geo -d $split_dir/1/ -o $split_dir/$i/
done

# slice
python3 -m scripts.make_slices -m $mask_path -o $DATA_DIR/expers/geographic/

# construct different folds
for i in $( seq 1 $n_folds)
    python3 -m scripts.geo.conf -i $i -t conf/geo/postprocess.yaml -o $split_dir/$i/postprocess.yaml
    python3 -m scripts.process_slices -o $split_dir/$i -m $DATA_DIR/expers/geographic/slices/slices.geojson -p $split_dir/$i/postprocess.yaml
done;
