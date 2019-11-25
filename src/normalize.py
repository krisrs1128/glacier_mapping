# normalize 
import os
import pickle

from torch.utils.data import DataLoader

from preprocess import online_mean_and_sd
from dataset import GlacierDataset


base_dir = '../data/toy_data'
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
