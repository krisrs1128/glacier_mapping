#!/usr/bin/env python
"""
To Run:

# run from glacier_mapping/ directory
python3 -m src.post_process.py
    --slice_dir=/path_to_glacier_slices/
    --slice_meta=/path_to_slice_metadata.geojson

"""
from addict import Dict
from argparse import ArgumentParser
from joblib import Parallel, delayed
from pathlib import Path
import geopandas as gpd
import numpy as np
import os
import src.data.process_slices_funs as pf
import yaml


if __name__ == "__main__":
    processed_dir = Path(os.environ["DATA_DIR"], "processed")
    conf_dir = Path(os.environ["ROOT_DIR"], "conf")

    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--conf",
        type=str,
        default=conf_dir / "postprocess.yaml",
        help="Path to the file specifying postprocessing options.",
    )
    parser.add_argument(
        "-d",
        "--slice_dir",
        type=str,
        default=processed_dir / "slices/",
        help="path to directory with all the slices",
    )
    parser.add_argument(
        "-m",
        "--slice_meta",
        type=str,
        default=processed_dir / "slices/slices_0-100.geojson",
        help="path to the slices metadata",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=processed_dir,
        help="path to output directory for postprocessed files",
    )
    parser.add_argument(
        "-n",
        "--n_cpu",
        type=int,
        default=4,
        help="Number of CPUs to parallelize processing over",
    )
    args = parser.parse_args()

    conf = Dict(yaml.safe_load(open(args.conf, "r")))
    slice_meta = gpd.read_file(Path(args.slice_meta))

    # filter all the slices to the ones that matter
    print("filtering")
    keep_ids = pf.filter_directory(
        slice_meta,
        filter_perc=conf.filter_percentage,
        filter_channel=conf.filter_channel,
    )

    # validation: get ids for the ones that will be training vs. testing.
    print("reshuffling")
    split_fun = getattr(pf, conf.split_method)
    split_ids = split_fun(keep_ids, conf.split_ratio, slice_meta=slice_meta)
    target_locs = pf.reshuffle(split_ids, args.output_dir)

    # global statistics: get the means and variances in the train split
    print("getting stats")
    conf.process_funs.normalize.stats_path = Path(
        conf.process_funs.normalize.stats_path
    )

    stats = pf.generate_stats(
        [p["img"] for p in target_locs["train"]],
        conf.normalization_sample_size,
        conf.process_funs.normalize.stats_path,
    )

    # postprocess individual images (all the splits)
    for split_type in target_locs:
        print(f"postprocessing {split_type}...")

        def wrapper(i):
            img, mask = pf.postprocess(
                target_locs[split_type][i]["img"],
                target_locs[split_type][i]["mask"],
                conf.process_funs,
            )

            np.save(target_locs[split_type][i]["img"], img)
            np.save(target_locs[split_type][i]["mask"], mask)

        para = Parallel(n_jobs=args.n_cpu)
        para(delayed(wrapper)(i) for i in range(len(target_locs[split_type])))
