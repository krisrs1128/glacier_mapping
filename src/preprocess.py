import logging
import os
from random import shuffle

import numpy as np
import pandas as pd

import geopandas
import rasterio

import utils

def get_sliced_mask(img, label, size=(512, 512), nan_value=0):
  mask = utils.get_mask(img, label, nan_value=nan_value)
  slices = utils.slice_image(mask, size=size)
  
  return mask, slices

def save_slice(data, save_loc, slice_type, img_name, num):
  slice_name = '_'.join([slice_type, img_name, 'slice', str(num)]) + '.npy'
  slice_path = os.path.join(save_loc, slice_name)
  np.save(slice_path, data)
  
  relative_path = os.path.join('slices', slice_name)
  
  return relative_path

def chunck_satelitte(img_path, labels, data_df, base_dir,
                     borders=None, basin=None, size=(512, 512), crop=True):
  img_name = os.path.splitext(os.path.basename(img_path))[0]
  logging.info('Image name :{} '.format(img_name))
  img = rasterio.open(img_path)
  
  # slice mask and original image
  img_np = np.moveaxis(img.read(), 0, 2)
  img_slices = utils.slice_image(img_np, size=size)
  _, mask_slices = get_sliced_mask(img, labels, size=size)
  
  # mask and slice borders if provided 
  if borders is not None:
      _, borders_slices = get_sliced_mask(img, borders, size=size, nan_value=1)
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
    data_dict['img_path'] = save_slice(filled_img_slice, save_loc, 'img', img_name, i)
    data_dict['mask_path'] = save_slice(mask_slices[i], save_loc, 'mask', img_name, i)
    
    # metadata
    is_nan = np.isnan(img_slice[:, :, 0])
    is_label = mask_slices[i] == 1
    
    data_dict['img_id'] = img_name
    data_dict['is_nan_perc'] = is_nan.sum() / is_nan.size
    data_dict['labels_perc'] = is_label.sum() / is_label.size
    data_dict['labeled_nan'] = (is_nan & is_label).sum() / is_label.sum()

    if borders is not None:
      is_border = borders_slices[i] == 1
      data_dict['border_path'] = save_slice(borders_slices[i], save_loc,
                                                             'borders', img_name, i)
      data_dict['in_border_perc'] = is_border.sum() / is_border.size
      data_dict['labels_in_border'] = (is_border & is_label).sum() / is_label.sum()
            
      if crop:
        filled_img_slice = np.nan_to_num(cropped_slices[i])
        data_dict['cropped_path'] = save_slice(filled_img_slice, save_loc,
                                                                'cropped_img', img_name, i)
        cropped_label = np.logical_and(mask_slices[i], borders_slices[i])
        data_dict['cropped_label'] = save_slice(cropped_label, save_loc,
                                                                'cropped_label', img_name, i)      
    
    if basin is not None:
      is_basin = basin_slices[i] == 1
      data_dict['basin_perc'] = is_basin.sum() / is_label.sum()  

    data_df = data_df.append(data_dict, ignore_index=True)

  return data_df 
    

def chunck_sat_files(sat_dir, labels_path, save_loc, borders_path=None, basin_path=None,
                     size=(512, 512)):
    labels = geopandas.read_file(labels_path)
    borders = geopandas.read_file(borders_path) if borders_path is not None else None
    basin = geopandas.read_file(basin_path) if basin_path is not None else None
  
    columns = ['img_id', 'img_path', 'mask_path', 'border_path',
               'is_nan_perc', 'labels_perc', 'labeled_nan', 'in_border_perc',
               'labels_in_border', 'basin_perc']
    sat_data = pd.DataFrame(columns=columns)

    files = os.listdir(sat_dir)
    n = len(files)
    for i, f in enumerate(files):
      logging.info('Processing file {}/{}.'.format(i + 1, n))
      img_path = os.path.join(sat_dir, f)
      img = rasterio.open(img_path)
      
      sat_data = chunck_satelitte(img_path, labels, sat_data, save_loc ,borders, basin, size=size)
    
    sat_data.to_csv(os.path.join(save_loc, 'sat_data.csv'), index=False)

def filter_images(sat_data_file, valid_cond_f, test_cond_f, save=True):
    sat_data = pd.read_csv(sat_data_file)
   
    sat_data['valid_data'] =  False
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
  sat_data = pd.read_csv(sat_data_file)
  
  n = len(sat_data[sat_data.train == 'train'])
  dev_size = int(n * perc)
  
  train_idx = list(sat_data[sat_data.train == 'train'].index)
  shuffle(train_idx)
  
  sat_data.loc[train_idx[:dev_size], 'train'] = label
  
  
  if not save:
    return sat_data
  else:
    sat_data.to_csv(sat_data_file, index=False)



