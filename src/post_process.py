#!/usr/bin/env python
from addict import Dict
from argparse import ArgumentParser
import addict
from pathlib import Path
import glob
import numpy as np
import os
import pandas as pd
import yaml
import postprocess_funs as pf

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--slice_dir", type=str, help="path to directory with all the slices")
    parser.add_argument("-m", "--slice_meta", type=str, help="path to the slices metadata")
    parser.add_argument("-o", "--output_dir", type=str, default="processed", help="path to output directory for postprocessed files")
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
    split_fun = getattr(pf, args.split_method)
    split_ids = split_fun(keep_ids, split_ratio, slices_meta)
    target_locs = pf.reshuffle(args.slice_dir, split_ids)

    # global statistics: get the means and variances in the train split
    train_img_paths = glob.glob(Path(args.output_dir, "train", "*img*"))
    stats = pf.generate_stats(
        train_img_paths,
        conf.normalization_sample_size,
        path(args.output_dir, "stats_train.json")
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
