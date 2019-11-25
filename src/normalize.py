# normalize 
import os
import pickle
from argparse import ArgumentParser

from torch.utils.data import DataLoader

from preprocess import online_mean_and_sd
from dataset import GlacierDataset

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--toy_data", default=True, type=bool,
                        help="whether to use the toy data or all the data")
    args = parser.parse_args()
    if args.toy_data:
        base_dir = '../data/toy_data'
    else:
        base_dir = '../data'

    data_file = 'sat_data.csv'
    borders = False
    use_snow_i = False

    # calculate mean and std for all channels

    train_dataset = GlacierDataset(base_dir, data_file, mode='train', borders=borders,
                                   use_snow_i=use_snow_i)
    channels, _, _ = train_dataset[0][0].shape

    normalization_loader = DataLoader(train_dataset)
    mean, std = online_mean_and_sd(normalization_loader, channels)

    norm_data = {'mean': mean, 'std':std}
    save_loc = os.path.join(base_dir, "normalization_data.pkl")
    pickle.dump(norm_data, open(save_loc, "wb"))
