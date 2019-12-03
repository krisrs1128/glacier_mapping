from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd

import src.utils as utils

def get_pseudo_debris_perc(row, base_folder):
    mask = np.load(base_folder / row.cropped_label)
    img = np.load(base_folder / row.cropped_path)
    img = np.moveaxis(img, -1, 0) # channels first
    debris_mask = utils.get_debris_glaciers(img, mask)
    return debris_mask.sum() / mask.sum()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--data_path", default="./data/sat_data.csv", type=str,
                        help="path of sat_data file to filter")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    df = pd.read_csv(data_path)
    debris_perc = df.apply(lambda row: get_pseudo_debris_perc(row, base_folder=data_path.parent),
                           axis=1)
    df['pseudo_debris_perc'] = debris_perc
    df.to_csv(data_path)
