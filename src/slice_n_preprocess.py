import os
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

import src.preprocess as preprocess
from src.utils import  load_conf


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument(
            "-c",
            "--conf_name",
            type=str,
            help="path to configuration file for slicing",
    )

    parsed_opts = parser.parse_args()
    config_values = load_conf(parsed_opts.conf_name)
    data_c = config_values["data"]
    valid_c = config_values["validity_values"]
    split_c = config_values["spliting_values"]

    base_dir = Path(data_c["data_path"])

    basin_path = base_dir / "vector_data/basin/Dudh_Koshi_Glacier.shp"

    labels_path = base_dir / f'vector_data/{data_c["year"]}/{data_c["country"]}/data/Glacier_{data_c["year"]}.shp'
    borders_path =  base_dir/ f'vector_data/borders/{data_c["country"]}/{data_c["country"]}.shp'
    sat_dir = base_dir / f'img_data/{data_c["year"]}/{data_c["country"]}'
    save_loc = base_dir / f'sat_files/{data_c["year"]}/{data_c["country"]}' 


    # slice all images in that folder
    preprocess.chunck_sat_files(sat_dir, labels_path, base_dir, save_loc,
                                borders_path=borders_path, basin_path=basin_path,
                                size=(data_c["size"], data_c["size"]),
                                year=data_c["year"], country=data_c["country"])

    def valid_cond_f(sat_data): return ((sat_data.labels_perc > valid_c["labels_perc"]) &
                                        (sat_data.labels_in_border > valid_c["labels_in_border"]) &
                                        (sat_data.is_nan_perc < valid_c["is_nan_perc"]))
    
    sat_data_file = os.path.join(save_loc, 'sat_data.csv')
    if split_c["random_test"]:
        def test_cond_f(sat_data): return pd.Series([False for _ in range(len(sat_data))])
        preprocess.filter_images(sat_data_file, valid_cond_f, test_cond_f)
        preprocess.split_train_test(
            sat_data_file, save=True, perc=split_c["test_perc"], label='test')

    else:
        def test_cond_f(sat_data): return (sat_data.basin_perc > 0)
        preprocess.filter_images(sat_data_file, valid_cond_f, test_cond_f)

    preprocess.split_train_test(sat_data_file, save=True, perc=split_c["dev_perc"])

    if data_c["merge"]:
        sat_path = base_dir / "sat_files"
        dfs = []
        for year_path in sat_path.iterdir():
            for country_path in year_path.iterdir():
                if (country_path / 'sat_data.csv').exists():
                    dfs.append(pd.read_csv(country_path / 'sat_data.csv'))
        pd.concat(dfs).to_csv(base_dir / 'sat_data.csv')