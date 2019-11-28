import logging
import os
import pickle
from argparse import ArgumentParser

import numpy as np
import pandas as pd

import utils

def get_debris_perc(row, base_folder):
    mask = np.load(os.path.join(base_folder, row.cropped_label))
    img = np.load(os.path.join(base_folder, row.cropped_path))
    img = np.moveaxis(img, -1, 0) # channels first
    debris_mask = utils.get_debris_glaciers(img, mask)
    return debris_mask.sum() / mask.sum()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--toy_data", default=True, type=bool,
                        help="whether to use the toy data or all the data")
    args = parser.parse_args()
    if args.toy_data:
        base_dir = '../data/toy_data'
    else:
        base_dir = '../data'

    df = pd.read_csv(os.path.join(base_dir, 'sat_data.csv'))
    debris_perc = df.apply(lambda row: get_debris_perc(row, base_folder=base_dir), axis=1)
    df['debris_perc'] = debris_perc
    df.to_csv(os.path.join(base_dir, 'sat_data.csv'))
