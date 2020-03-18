# This code calculates the IOU between debris labels and pseudo debris labels 
# for each slice of the specified image filename.
# Usage: python3 test_calculate_iou.py -f [shp_filename with debris information] -p [pattern for corresponding landsat image filename]
# python3 test_calculate_iou.py -f ../data/vector_data/2005/nepal/data/Glacier_2005.shp -p LE07_140041_*
# Output:   The real debris labels are saved in ./temp_files/real/filename
#           The snow_index debris labels are saved in ./temp_files/snow/filename 
import sys
sys.path.append('../')
import rasterio
import geopandas
import argparse
import glob
import os
import numpy as np

import torchvision.transforms as T
from src.utils import get_debris_glaciers
from src.utils import get_mask
from src.utils import slice_image
from src.preprocess import save_slice
import matplotlib 

def get_iou(snow,real):
    _union = np.zeros_like(snow)
    _intersection = np.zeros_like(snow)
    _union[(snow==1)|(real==1)] = 1
    _intersection[(snow==1)&(real==1)] = 1
    return np.sum(_union), np.sum(_intersection)

if __name__ == "__main__":

    threshhold = 0.15       # Threshold for snow index

    parser = argparse.ArgumentParser()
    parser.add_argument(
            "-f",
            "--input_shp_file",
            type=str,
            help="Define input shp file that contains clean ice and debris glaciers",
    )
    parser.add_argument(
            "-p",
            "--name_pattern",
            type=str,
            help="pattern for tiff files (Default: LE07_140041_20051012*)",
    )

    parsed_opts = parser.parse_args()
    shp_file = parsed_opts.input_shp_file
    name_pattern = parsed_opts.name_pattern
    try:
        assert(shp_file)
    except Exception as e:
        print("Input filename must be specified. Use flag -f")
        exit(0)
    try:
        assert(name_pattern)
    except Exception as e:
        name_pattern = "LE07_140041_20051012*"
        print(e," Using default pattern LE07_140041_20051012*")

    labels = geopandas.read_file(shp_file)
    try:
        sat_image = rasterio.open(glob.glob('../data/img_data/2005/nepal/'+name_pattern)[0])
    except Exception as e:
        print("File "+name_pattern+" not found")
        exit(0)
    real_clean = labels[labels["Glaciers"] == "Clean Ice"]
    real_debris = labels[labels["Glaciers"] == "Debris covered"]

    slice_path = "../data/slices/img_"+name_pattern
    mask_path = "../data/slices/cropped_label_"+name_pattern
    real_debris_mask_path = "../data/slices/actual_debris_cropped_label_"+name_pattern

    img_slices = sorted(glob.glob(slice_path))
    mask_slices = sorted(glob.glob(mask_path))
    real_debris_mask_slices = sorted(glob.glob(real_debris_mask_path))
    # real_debris_mask = get_mask(sat_image,real_debris)
    # real_debris_mask_slices = slice_image(real_debris_mask)

    intersection, union = 1,1

    for image_path, mask_path, actual_debris_mask_path in zip(img_slices,mask_slices,real_debris_mask_slices):
        img = np.load(image_path)
        img = T.ToTensor()(img)
        mask = np.load(mask_path)
        snow_i_debris = get_debris_glaciers(img, mask, thresh=threshhold)
        actual_debris_mask = np.load(actual_debris_mask_path)
        _filename = image_path.split("/")[-1].split(".")[0]
        # index = int(_filename.split("_")[-1])

        if(np.any(snow_i_debris) or np.any(actual_debris_mask)):
            _union, _intersection = get_iou(snow_i_debris,actual_debris_mask)
            print("Filename: ",_filename,"\tIOU= ",_intersection/_union)
            intersection += _intersection
            union += _union

        if np.any(snow_i_debris):
            if not os.path.exists('./temp_files'):
                os.makedirs('./temp_files')
            if not os.path.exists('./temp_files/snow'):
                os.makedirs('./temp_files/snow')
            matplotlib.image.imsave('./temp_files/snow/'+_filename+'.jpeg',snow_i_debris)
        if np.any(actual_debris_mask):
            if not os.path.exists('./temp_files'):
                os.makedirs('./temp_files')
            if not os.path.exists('./temp_files/real'):
                os.makedirs('./temp_files/real')    
            matplotlib.image.imsave('./temp_files/real/'+_filename+'.jpeg',actual_debris_mask)
    

    print("Cumulative IOU \tIntersection: ",intersection,"\tUnion: ",union,"\tIOU= ",intersection/union)