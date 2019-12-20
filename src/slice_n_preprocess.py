import os
import itertools
import logging
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

import src.preprocess as preprocess
from src.utils import  load_conf
from src.cluster_utils import env_to_path

def preprocess_country(base_dir, test_basin_path, dev_basin_path,
                       country, year, data_c, valid_c, split_c):
    labels_path = base_dir / f'vector_data/{year}/{country}/data/Glacier_{year}.shp'
    borders_path =  base_dir/ f'vector_data/borders/{country}/{country}.shp'
    sat_dir = base_dir / f'img_data/{year}/{country}'
    save_loc = Path(base_dir, f'sat_files/{year}/{country}')
    save_loc.mkdir(parents=True, exist_ok=True)

    # slice all images in that folder
    preprocess.chunck_sat_files(sat_dir, labels_path, base_dir, save_loc,
                                borders_path=borders_path, test_basin_path=test_basin_path,
                                dev_basin_path=dev_basin_path,
                                size=(data_c["size"], data_c["size"]),
                                year=year, country=country)

    def valid_cond_f(sat_data): return ((sat_data.labels_perc > valid_c["labels_perc"]) &
                                        (sat_data.labels_in_border > valid_c["labels_in_border"]) &
                                        (sat_data.is_nan_perc < valid_c["is_nan_perc"]))

    sat_data_file =  Path(save_loc, 'sat_data.csv')

    if split_c["random_split"]:
        def random_cond_f(sat_data): return pd.Series([False for _ in range(len(sat_data))])
        preprocess.filter_images(sat_data_file, valid_cond_f, random_cond_f, random_cond_f)
        preprocess.split_train_test(
            sat_data_file, save=True, perc=split_c["test_perc"], label='test')
        preprocess.split_train_test(
            sat_data_file, save=True, perc=split_c["dev_perc"], label='dev')
    else:
        def test_cond_f(sat_data): return (sat_data.test_basin_perc > 0)
        def dev_cond_f(sat_data): return (sat_data.dev_basin_perc > 0)
        preprocess.filter_images(sat_data_file, valid_cond_f, test_cond_f, dev_cond_f)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument(
            "-c",
            "--conf_name",
            type=str,
            help="path to configuration file for slicing",
    )
    logging.getLogger().setLevel(logging.INFO)


    parsed_opts = parser.parse_args()
    config_values = load_conf(parsed_opts.conf_name)
    data_c = config_values["data"]
    valid_c = config_values["validity_values"]
    split_c = config_values["spliting_values"]

    base_dir = Path(env_to_path(data_c["data_path"]))

    test_basin_path = base_dir / "vector_data/basin/Dudh_Koshi_Glacier.shp"
    dev_basin_path = base_dir / "vector_data/val/val.shp"

    countries = data_c["country"]
    years = data_c["year"]


    if not years:
        years = [year_path.name for year_path in Path(base_dir, 'img_data').iterdir()]

    if not countries:
        countries = []
        for year in years:
            year_path = Path(base_dir, 'img_data', year)
            for country_path in year_path.iterdir():
                countries.append(country_path.name)
        countries = set(countries)
    
    for year, country in list(itertools.product(years, countries)):
        if Path(base_dir, f'img_data/{year}/{country}').exists():
            logging.info(f'Processing {year}/{country}')
            preprocess_country(base_dir, test_basin_path, dev_basin_path,
                               country, year, data_c, valid_c, split_c)
            
        sat_path = base_dir / "sat_files"
        dfs = []
        for year_path in sat_path.iterdir():
            for country_path in year_path.iterdir():
                if (country_path / 'sat_data.csv').exists():
                    dfs.append(pd.read_csv(country_path / 'sat_data.csv'))
        pd.concat(dfs).to_csv(base_dir / 'sat_data.csv')
