#!/usr/bin/env python
import argparse
import pathlib
import yaml

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Make config file for masking")
    parser.add_argument("-i", "--input_dir", type=str)
    parser.add_argument("-o", "--output_file", type=str, default="mask.yaml")
    parser.add_argument("-m", "--mask_path", type=str, default="/mnt/blobfuse/glaciers/raw/vector_data/2005/hkh/Glacier_2005.shp")
    args = parser.parse_args()

    mask_conf = {}
    input_tiles = pathlib.Path(args.input_dir).glob("*.tif*")
    for i, path in enumerate(input_tiles):
        mask_conf[f"mask_{i}"] = {
            "img_path": str(path.resolve()),
            "mask_paths": args.mask_path.split(",")
        }

    with open(args.output_file, 'w') as f:
        yaml.dump(mask_conf, f, default_flow_style=False)
