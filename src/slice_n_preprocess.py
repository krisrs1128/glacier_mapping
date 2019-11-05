import os
from argparse import ArgumentParser

import preprocess

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--toy_data", default=True, type=bool,
                        help="whether to use the toy data or all the data")
    args = parser.parse_args()
    if args.toy_data:
        base_dir = '../data/toy_data'
        size = (128, 128)
    else:
        base_dir = '../data'
        size = (512, 512)

    # TODO: use conf file for the pathes
    labels_path = os.path.join(
        base_dir, 'vector_data/nepal_2010/data/Glacier_2010.shp')
    borders_path = os.path.join(
        base_dir, 'vector_data/borders/nepal_boundries_detailed/nepal_boundries_1.shp')
    basin_path = os.path.join(
        base_dir, 'vector_data/basin/Dudh_Koshi_Glacier.shp')
    sat_dir = os.path.join(base_dir, 'img_data')
    preprocess.chunck_sat_files(sat_dir, labels_path, base_dir,
                                borders_path=borders_path, basin_path=basin_path,
                                size=size)

    def valid_cond_f(sat_data): return ((sat_data.labels_perc > 0) &
                                        (sat_data.labels_in_border > 0) &
                                        (sat_data.is_nan_perc < 0.9))
    sat_data_file = os.path.join(base_dir, 'sat_data.csv')
    if args.toy_data:
        def test_cond_f(sat_data): return (sat_data.basin_perc < 0)
        preprocess.filter_images(sat_data_file, valid_cond_f, test_cond_f)
        preprocess.split_train_test(
            sat_data_file, save=True, perc=0.1, label='test')

    else:
        def test_cond_f(sat_data): return (sat_data.basin_perc > 0)
        preprocess.filter_images(sat_data_file, valid_cond_f, test_cond_f)

    preprocess.split_train_test(sat_data_file, save=True)
