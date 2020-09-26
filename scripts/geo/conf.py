#!/usr/bin/env python
import argparse
import yaml

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build dummy configuration file for run i of geographic experiment")
    parser.add_argument("-o", "--output_file", type=str)
    parser.add_argument("-i", "--index_split", type=int, default = 1)
    parser.add_argument("-t", "--template_conf", type=str, default = "conf/process_geo.conf")
    args = parser.parse_args()

    pconf = yaml.safe_load(open(args.template_conf, "r"))
    train_path = pconf["split_method"]["geographic_split"]["geojsons"]["train"]
    test_path = pconf["split_method"]["geographic_split"]["geojsons"]["test"]
    stats_path = pconf["process_funs"]["normalize"]["stats_path"]

    pconf["split_method"]["geographic_split"]["geojsons"]["train"] = train_path.replace("FOLD_NUM", str(args.index_split))
    pconf["split_method"]["geographic_split"]["geojsons"]["test"] = test_path.replace("FOLD_NUM", str(args.index_split))
    pconf["process_funs"]["normalize"]["stats_path"] = stats_path.replace("FOLD_NUM", str(args.index_split))

    with open(args.output_file, 'w') as f:
        yaml.dump(pconf, f, default_flow_style=False)
