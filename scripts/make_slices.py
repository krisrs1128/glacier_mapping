#!/usr/bin/env python
"""Preprocessing script from colab notebook

python3 -m make_slices -m ../conf/geo/mask.yaml -o $DATA_DIR/expers/geographic/ -p ../conf/geo/postprocess.yaml
"""
from addict import Dict
from glacier_mapping.data.mask import generate_masks
from glacier_mapping.data.slice import write_pair_slices
import argparse
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
    args = parser.parse_args()

    # generate masks
    masking_paths = yaml.safe_load(open(args.masking_yaml))
    img_paths = [p["img_path"] for p in masking_paths.values()]
    mask_paths = [p["mask_paths"] for p in masking_paths.values()]
    if "border_path" in next(iter(masking_paths)):
        border_paths = [p["border_path"] for p in masking_paths.values()]
    else:
        border_paths = []

    output_dir = pathlib.Path(args.output_dir)

    generate_masks(img_paths, mask_paths,
        border_paths=border_paths, out_dir=output_dir / "masks/")
    paths = pd.read_csv(output_dir / "masks" / "mask_metadata.csv")
    slice_dir = output_dir / "slices"
    slice_dir.mkdir(parents=True, exist_ok=True)

    metadata = []
    for row in range(len(paths)):
        print(f"## Slicing tiff {row +1}/{len(paths)} ...")
        metadata_ = write_pair_slices(
            paths.iloc[row]["img"],
            paths.iloc[row]["mask"],
            slice_dir,
            border_path=paths.iloc[row]["border"],
            out_base=f"slice_{paths.index[row]}"
        )
        metadata.append(metadata_)

    metadata = pd.concat(metadata, axis=0)
    out_path = pathlib.Path(slice_dir, "slices.geojson")
    metadata.to_file(out_path, index=False, driver="GeoJSON")
