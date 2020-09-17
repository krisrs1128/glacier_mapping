#!/usr/bin/env python
"""Preprocessing script from colab notebook

python3 -m experiment_helpers.preprocess -m conf/geo/mask.yaml -o $DATA_DIR/expers/geographic/splits/01/ -p /conf/geo/postprocess.yaml
"""
from addict import Dict
from glacier_mapping.data.mask import generate_masks
from glacier_mapping.data.slice import write_pair_slices
import geopandas as gpd
import glacier_mapping.data.process_slices_funs as pf
import numpy as np
import pandas as pd
import pathlib
import yaml

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess raw tiffs into slices")
    parser.add_argument("-m", "--masking_yaml", type=str, default = "conf/masks_geo_exper.conf")
    parser.add_argument("-o", "--output_dir", type=str)
    parser.add_argument("-p", "--postprocess_conf", type=str, default = "conf/process_geo.conf")
    args = parser.parse_args()

    # generate masks
    masking_paths = yaml.load(open(args.masking_yaml))
    img_paths = [p["img_path"] for p in masking_paths.values()]
    mask_paths = [p["mask_paths"] for p in masking_paths.values()]
    generate_masks(img_paths, mask_paths)

    output_dir = pathlib.Path(args.output_dir)
    paths = pd.read_csv(processed_dir / "masks" / "mask_metadata.csv")
    slice_dir = processed_dir / "slices"
    slice_dir.mkdir(parents=True, exist_ok=True)

    metadata = []
    for row in range(len(paths)):
        print(f"## Slicing tiff {row +1}/{len(paths)} ...")
        metadata_ = write_pair_slices(
            paths.iloc[row]["img"],
            paths.iloc[row]["mask"],
            slice_dir,
            f"slice_{paths.index[row]}"
        )
        metadata.append(metadata_)

    metadata = pd.concat(metadata, axis=0)
    out_path = pathlib.Path(slice_dir, "slices.geojson")
    metadata.to_file(out_path, index=False, driver="GeoJSON")

    pconf = Dict(yaml.safe_load(open(args.postprocess_conf, "r")))
    slice_meta = gpd.read_file(slice_dir / "slices.geojson")

    # filter all the slices to the ones that matter
    print("filtering")
    keep_ids = pf.filter_directory(
        slice_meta,
        filter_perc=pconf.filter_percentage,
        filter_channel=pconf.filter_channel,
    )

    # validation: get ids for the ones that will be training vs. testing.
    print("reshuffling")
    split_fun = getattr(pf, pconf.split_method)
    split_ids = split_fun(keep_ids, pconf.split_ratio, slice_meta=slice_meta)
    target_locs = pf.reshuffle(split_ids, output_dir)

    # global statistics: get the means and variances in the train split
    print("getting stats")
    pconf.process_funs.normalize.stats_path = processed_dir / \
        pathlib.Path(pconf.process_funs.normalize.stats_path)

    stats = pf.generate_stats(
        [p["img"] for p in target_locs["train"]],
        pconf.normalization_sample_size,
        pconf.process_funs.normalize.stats_path,
    )

    # postprocess individual images (all the splits)
    for split_type in target_locs:
        print(f"postprocessing {split_type}...")
        for i in range(len(target_locs[split_type])):
            img, mask = pf.postprocess(
                target_locs[split_type][i]["img"],
                target_locs[split_type][i]["mask"],
                pconf.process_funs,
            )

            np.save(target_locs[split_type][i]["img"], img)
            np.save(target_locs[split_type][i]["mask"], mask)
