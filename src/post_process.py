#!/usr/bin/env python
from addict import Dict
from argparse import ArgumentParser
from pathlib import Path
import addict
import numpy as np
import pandas as pd
import postprocess_funs as pf
import yaml


"""
To Run:

python3 src/post_process.py 
    --slice_dir=/path_to_glacier_slices/ 
    --slice_meta=/path_to_slice_metadata.csv

example: 
    python3 src/post_process.py --slice_dir=/scratch/akera/glaciers_slices/ --slice_meta=/scratch/akera/glacier_slices/slice_metadata.csv

"""

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--slice_dir", type=str, help="path to directory with all the slices")
    parser.add_argument("-m", "--slice_meta", type=str, help="path to the slices metadata")
    parser.add_argument("-o", "--output_dir", type=str, default="./processed", help="path to output directory for postprocessed files")
    parser.add_argument("-c", "--conf", type=str, default="conf/postprocess.yaml", help="Path to the file specifying postprocessing options.")
    args = parser.parse_args()

    conf = Dict(yaml.safe_load(open(args.conf, "r")))
    slice_meta = pd.read_csv(Path(args.slice_meta))

    # filter all the slices to the ones that matter
    keep_ids = pf.filter_directory(
        slice_meta,
        filter_perc=conf.filter_percentage,
        filter_channel=conf.filter_channel
    )

    # validation: get ids for the ones that will be training vs. testing.
    split_fun = getattr(pf, conf.split_method)
    split_ids = split_fun(keep_ids, conf.split_ratio, slice_meta=slice_meta)
    target_locs = pf.reshuffle(split_ids, args.output_dir)

    # global statistics: get the means and variances in the train split
    stats = pf.generate_stats(
        [p["img"] for p in target_locs["train"]],
        conf.normalization_sample_size,
        Path(args.output_dir, "stats_train.json")
    )

    # postprocess individual images (all the splits)
    for split_type in target_locs:
        for i in range(len(target_locs[split_type])):
            funs_seq = [getattr(pf, f) for f in conf.funs]
            pf.postprocess(
                target_locs[split_type][i]["img"],
                target_locs[split_type][i]["mask"],
                funs_seq,
                stats=stats
            )
