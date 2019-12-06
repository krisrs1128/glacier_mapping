#!/usr/bin/env python
import logging
import os
import pathlib
import pickle
import random

import numpy as np
import pandas as pd
import geopandas
import rasterio
import torch
import torchvision.transforms as T

import src.utils as utils


def get_sliced_mask(img, label, size=(512, 512), nan_value=0):
    """Given an image and label, return the mask and the its slices.
    Args:
        raster_img (rasterio dataset object): the rater image to mask
        vector_data (iterable polygon data): the labels to mask according to
        size (int, int): size of the resulting slices
        nan_value (int): the value to fill nan areas with
    Returns:
        mask (numpy.array): a binary mask of img according to labels
        slices [numpy.array]: list of numpy slices of the mask
    """

    mask = utils.get_mask(img, label, nan_value=nan_value)
    slices = utils.slice_image(mask, size=size)

    return mask, slices


def save_slice(data, save_loc, slice_type, img_name, num):
    """Construct the path to save and save a given slice.
    Args:
        data (numpy.array): the slice to save
        save_loc (str): the base folder to save all slices
        slice_type (str): type of slice, like mask, img or cropped_img
        img_name (str): the id of the image the slice belongs to
        num (int): the number of the slice, relative to all the slices in the same image
    Returns:
        relative_path (str): the saving location of the slice relative to the base directory"""

    slice_name = '_'.join([slice_type, img_name, 'slice', str(num)]) + '.npy'
    slice_path = os.path.join(save_loc, slice_name)
    np.save(slice_path, data)

    relative_path = os.path.join('slices', slice_name)

    return relative_path


def chunck_satelitte(img_path, labels, data_df, base_dir,
                     borders=None, basin=None, size=(512, 512), crop=True,
                     country=None, year=None):
    """Chunck a given satelitte image with the related labels and save metadata.
    Args:
        img_path (str): the path to the image to chunck
        labels (iterable polygon data): the labels of interest to mask with
        data_df (pandas.Dataframe): the metadata dataframe to use
        base_dir (str): the base directory to save the slices in
        borders (iterable polygon data): borders data to mask
        basin (iterable polygon data): basin of interest data to mask
        size (int, int): the size to slice with
        crop (bool): whether to crop areas outside of borders
        country (str): the name of the country the image belongs to, for metadata
        year (str): the year the image belongs to, for metadata
    Returns:
        pandas.Dataframe: a dataframe with all the realtive pathes and metadata"""

    img_name = os.path.splitext(os.path.basename(img_path))[0]
    logging.info('Image name :{} '.format(img_name))
    img = rasterio.open(img_path)

    # slice mask and original image
    img_np = np.moveaxis(img.read(), 0, 2)
    img_slices = utils.slice_image(img_np, size=size)
    _, mask_slices = get_sliced_mask(img, labels, size=size)

    # mask and slice borders if provided
    if borders is not None:
        _, borders_slices = get_sliced_mask(
            img, borders, size=size, nan_value=1)
        if crop:
            cropped_img = utils.crop_raster(img, borders)
            cropped_img = np.moveaxis(cropped_img, 0, 2)
            cropped_slices = utils.slice_image(cropped_img, size=size)

    # mask and slice basin if provided
    if basin is not None:
        _, basin_slices = get_sliced_mask(img, basin, size=size)

    # save slices and fill metadata
    for i, img_slice in enumerate(img_slices):
        data_dict = {}

        # paths
        save_loc = os.path.join(base_dir, 'slices')
        if not os.path.exists(save_loc):
            os.mkdir(save_loc)

        # save valid images only
        filled_img_slice = np.nan_to_num(img_slice)
        data_dict['img_path'] = save_slice(
            filled_img_slice, save_loc, 'img', img_name, i)
        data_dict['mask_path'] = save_slice(
            mask_slices[i], save_loc, 'mask', img_name, i)

        # metadata
        is_nan = np.isnan(img_slice[:, :, 0])
        is_label = mask_slices[i] == 1

        data_dict['img_id'] = img_name
        data_dict['country'] = country
        data_dict['year'] = year
        data_dict['is_nan_perc'] = is_nan.sum() / is_nan.size
        data_dict['labels_perc'] = is_label.sum() / is_label.size
        data_dict['labeled_nan'] = (is_nan & is_label).sum() / is_label.sum()

        if borders is not None:
            is_border = borders_slices[i] == 1
            data_dict['border_path'] = save_slice(borders_slices[i], save_loc,
                                                  'borders', img_name, i)
            data_dict['in_border_perc'] = is_border.sum() / is_border.size
            data_dict['labels_in_border'] = (
                is_border & is_label).sum() / is_label.sum()

            if crop:
                filled_img_slice = np.nan_to_num(cropped_slices[i])
                data_dict['cropped_path'] = save_slice(filled_img_slice, save_loc,
                                                       'cropped_img', img_name, i)
                cropped_label = np.logical_and(
                    mask_slices[i], borders_slices[i])
                data_dict['cropped_label'] = save_slice(cropped_label, save_loc,
                                                        'cropped_label', img_name, i)

        if basin is not None:
            is_basin = basin_slices[i] == 1
            data_dict['basin_perc'] = is_basin.sum() / is_label.sum()

        data_df = data_df.append(data_dict, ignore_index=True)

    return data_df


def chunck_sat_files(sat_dir, labels_path, save_loc, df_loc, borders_path=None,
                     basin_path=None, size=(512, 512), year=None, country=None):
    """Chunck all the images in a folder and construct their metadata.
    Args:
        sat_dir (str): the path of the directory
        labels_path (str): the path to the labels to mask with
        save_loc (str): where to save the data after chuncking
        df_loc (str): the location of a dataframe to save the metadata
        basin_path (str): the path to a basin of interst for test labels
        size (int, int): the size of resulting slices
        year (str): the year the images belong to
        country (str): the country the images belong to
    Returns: None
    """

    labels = geopandas.read_file(labels_path)
    borders = geopandas.read_file(
        borders_path) if borders_path is not None else None
    basin = geopandas.read_file(basin_path) if basin_path is not None else None

    columns = ['img_id', 'year', 'country', 'img_path', 'mask_path', 'border_path',
               'is_nan_perc', 'labels_perc', 'labeled_nan', 'in_border_perc',
               'labels_in_border', 'basin_perc']
    sat_data = pd.DataFrame(columns=columns)

    files = os.listdir(sat_dir)
    n = len(files)

    for i, f in enumerate(files):
        logging.info('Processing file {}/{}.'.format(i + 1, n))
        
        img_path = os.path.join(sat_dir, f)
        sat_data = chunck_satelitte(
            img_path, labels, sat_data, save_loc, borders=borders, basin=basin,
            size=size, year=year, country=country)

    sat_data.to_csv(os.path.join(df_loc, 'sat_data.csv'), index=False)

def filter_images(sat_data_file, valid_cond_f, test_cond_f, save=True):
    """filter image according to metadata.
    Args:
        sat_data_file (str): path to satlitte imagery metadata
        valid_cond_f (function): function that returns if a slice is a valid slice or should be ignored
        valid_cond_f (function): function that returns if a slice is a test slice
        save (bool): whether to save the resulting dataframe
    Returns:
        None or pd.Dataframe (metadata with column for train/test labels)
    """
    sat_data = pd.read_csv(sat_data_file)
    sat_data['valid_data'] = False
    sat_data.loc[valid_cond_f(sat_data), 'valid_data'] = True
    sat_data['train'] = 'invalid'
    sat_data.loc[sat_data.valid_data, 'train'] = 'train'
    test_cond = test_cond_f(sat_data) & (sat_data.valid_data)
    sat_data.loc[test_cond, 'train'] = 'test'

    if not save:
        return sat_data
    else:
        sat_data.to_csv(sat_data_file, index=False)


def split_train_test(sat_data_file, perc=0.2, save=True, label='dev'):
    """Split satelitte metadat into train and dev/test.
    Args:
        sat_data_file (str): path to metadata file
        perc (float): the percentage of the dev/test in data
        save (bool): whether to save the resulting dataframe
        label (str): the name of the minority label (should be dev or test)
    Returns:
        None or pd.Dataframe (metadata with column for dev/test labels)"""
    sat_data = pd.read_csv(sat_data_file)

    n = len(sat_data[sat_data.train == 'train'])
    dev_size = int(n * perc)

    train_idx = list(sat_data[sat_data.train == 'train'].index)
    random.seed(0)
    random.shuffle(train_idx)

    sat_data.loc[train_idx[:dev_size], 'train'] = label

    if not save:
        return sat_data
    else:
        sat_data.to_csv(sat_data_file, index=False)


def online_mean_and_sd(loader, channels):
    """Compute the mean and sd in an online fashion.
    Var[x] = E[X^2] - E^2[X]
    Args:
        loader (torch.data.utils.DataLoader): the training dataloader to get normalization from
        channels (int): how many channels to use
    Returns:
        (torch.tensor, torch.tensor): (mean, std)"""

    cnt = 0
    fst_moment = torch.empty(channels)
    snd_moment = torch.empty(channels)

    for img, _ in loader:
        b, _, h, w = img.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(img, dim=[0, 2, 3])
        sum_of_square = torch.sum(img ** 2, dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

        cnt += nb_pixels

    return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)


def get_normalization(data_config):
    """Returns the normalization transformation to use according to configuration
    Args:
        data_config (dict): configuration of the data
    Returns:
        torchvision.transforms"""
        
    norm_data_file = pathlib.Path(data_config.path, "normalization_data.pkl")
    norm_data = pickle.load(open(norm_data_file, "rb"))
    mean, std = norm_data["mean"], norm_data["std"]

    if data_config.use_snow_i: data_config.channels_to_inc.append(12)
    if data_config.border: data_config.channels_to_inc.append(13)
    
    channels_mean = [mean[i] for i in data_config.channels_to_inc]
    channels_std = [std[i] for i in data_config.channels_to_inc] 

    # if data_config.use_snow_i:
    #     channels_mean.append(mean[12])
    #     channels_std.append(std[12])

    # if data_config.border:
    #     channels_mean.append(mean[13])
    #     channels_std.append(std[13])

    img_transform = T.Normalize(channels_mean, channels_std)

    return img_transform
