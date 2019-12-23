# normalize 
import pickle
from argparse import ArgumentParser
from pathlib import Path

from torch.utils.data import DataLoader

from src.preprocess import online_mean_and_sd
from src.dataset import GlacierDataset

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--base_dir", default="./data", type=str,
                        help="path of base directory that has all the data")
    parser.add_argument("--data_file", default="sat_data.csv", type=str,
                        help="filtered sat_data to normalize")
    parser.add_argument("--normalization_data", default="normalization_data.pkl", type=str,
                        help="name of the normalization_data")
    args = parser.parse_args()
    base_dir = Path(args.base_dir)

    data_file = args.data_file

    # calculate mean and std for all channels
    train_dataset = GlacierDataset(base_dir, data_file, mode='train',
                                   use_snow_i=True, borders=True)
    channels, _, _ = train_dataset[0][0].shape

    normalization_loader = DataLoader(train_dataset)
    mean, std = online_mean_and_sd(normalization_loader, channels)

    norm_data = {'mean': mean, 'std':std}
    save_loc = Path(base_dir, args.normalization_data)
    pickle.dump(norm_data, open(save_loc, "wb"))
