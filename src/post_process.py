import pandas as pd
import numpy as np
import addict
import postprocess_funs
from addict import Dict
import pdb
import glob
import os
from argparse import ArgumentParser

if __name__ == '__main__':

    # inputs:
    #    - directory with sliced images
    #    - csv with the slices metadata
    #    - postprocessing specific parameters (e.g., filter percentage, validation method)
    #
    # outputs:
    #    - transformed images in their correct directories

    parser = ArgumentParser()
    parser.add_argument("-d", "--slice_dir", type=str, help="path to directory with all the slices")
    parser.add_argument("-m", "--slice_meta", type=str, help="path to the slices metadata")
    parser.add_argument("-o", "--output_dir", type=str, default="processed", help="path to output directory for postprocessed files")
    parser.add_argument("-c", "--conf", type=str, help="Path to the file specifying postprocessing options.")
    args = parser.parse_args()

    conf = Dict(yaml.load(args.conf))
    slice_meta = pd.DataFrame(pathlib.Path(args.slice_meta))

    # filter all the slices to the ones that matter
    keep_ids = filter_directory(
        args.slices_meta,
        filter_perc=conf.filter_percentage,
        filter_channel=conf.filter_channel
    )

    # validation: get ids for the ones that will be training vs. testing.
    split_fun = getattr(postprocess_funs, args.split_method)
    split_ids = split_fun(keep_ids, split_ratio, slices_meta)
    target_locs = postprocess_funs.reshuffle(args.slice_dir, split_ids)

    # global statistics: get the means and variances in the train split
    train_img_paths = glob.glob(Path(args.output_dir, "train", "*img*"))
    stats = generate_stats(
        train_img_paths,
        conf.normalization_sample_size,
        path(args.output_dir, "stats_train.json")
    )

    # postprocess individual images (all the splits)
    for split_type in target_locs:
        for i in range(len(target_locs[split_type])):
            postprocess(
                target_locs[split_type][i]["img"],
                target_locs[split_type][i]["mask"],
                [normalize, remove_nas, ...]
            )
