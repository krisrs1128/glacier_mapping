# This code is used to plot the band intensity for each of the classes
# x-mean(x)/sd(x)
# Usage: python3 test_get_clean_debris.py -f [shp_filename with debris information] -p [pattern for shp_filenames]
# python3 test_plot_band_intensity.py -f ../data/vector_data/2005/nepal/data/Glacier_2005.shp -p LE07_140041_20051012*
# Output:   The output is saved as test.png in the current directory
import sys
sys.path.append('../')
import rasterio
import geopandas
import argparse
import pickle
import glob
import numpy as np
import pandas as pd
import rasterio

import torchvision.transforms as T
import matplotlib 
import matplotlib.pyplot as plt
from rasterio.mask import mask as rasterio_mask
from sklearn.preprocessing import Normalizer
from random import randint

np.random.seed(7)

def check_condition(x):
    if not np.any(x[:-2]):              # Discard all 0 values   
        return False
    if x[9] in [752.0, 756.0, 760.0, 764.0]:    # Cloud High
        return False
    if x[9] in [928.0, 932.0, 936, 940, 960, 964, 968, 972]:    # Cloud Shadow High
        return False
    if x[10] < 4390:                # Only use 1% values with elevation < 4360 
        return randint(0,99) % 100 == 0
    if x[6] < 83:
        return randint(0,9) % 10 == 0
    if not float(x[9]).is_integer():           # Extrapolated values in GEE
        return randint(0,19) % 20 == 0
    return True

if __name__ == "__main__":
    n_points = -1

    norm_data_file = "../data/normalization_data.pkl"
    norm_data = pickle.load(open(norm_data_file, "rb"))
    mean, std = norm_data["mean"], norm_data["std"]
    norm_mean, norm_std = mean.numpy(), std.numpy()
    norm_mean = np.append(norm_mean[0:5],norm_mean[5])
    norm_std = np.append(norm_std[0:5],norm_std[5])

    parser = argparse.ArgumentParser()
    parser.add_argument(
            "-f",
            "--input_shp_file",
            type=str,
            help="Define input shp file that contains clean ice and debris glaciers",
    )
    parser.add_argument(
            "-n",
            "--max_points_per_label",
            type=str,
            help="Maximum number of pixels per label class",
    )
    parser.add_argument(
            "-p",
            "--name_pattern",
            type=str,
            help="pattern for tiff files (Default: LE07_140041_20051012*)",
    )

    parser.add_argument(
            "-o",
            "--output_filename",
            type=str,
            help="name of output file (Default: test.png)",
    )
    
    parsed_opts = parser.parse_args()
    shp_file = parsed_opts.input_shp_file
    name_pattern = parsed_opts.name_pattern
    max_points_per_label = parsed_opts.max_points_per_label
    try:
        assert(shp_file)
    except Exception as e:
        print("Input filename must be specified. Use flag -f")
        exit(0)
    try:
        assert(max_points_per_label)
    except Exception as e:
        max_points_per_label = 15000
        print("Using default number of points 15000")
    try:
        assert(name_pattern)
    except Exception as e:
        name_pattern = "LE07_140041_20051012*"
        print(e," Using default pattern LE07_140041_20051012*")
    try:
        assert(output_filename)
    except Exception as e:
        output_filename = "test.png"
        print(e," Using default filename test.png")

    labels = geopandas.read_file(shp_file)
    try:
        sat_image = rasterio.open(glob.glob('../data/img_data/2005/nepal/'+name_pattern)[0])
    except Exception as e:
        print("File "+name_pattern+" not found")
        exit(0)
    clean = labels[labels["Glaciers"] == "Clean Ice"]
    debris = labels[labels["Glaciers"] == "Debris covered"]

    image_path = "../data/img_data/2005/nepal/"+name_pattern
    df = pd.DataFrame(columns=["B1","B2","B3","B4","B5","B6_VCID_1","B6_VCID_2","B7","B8","BQA","elevation","slope","label"])
    images = sorted(glob.glob(image_path))

    x = np.arange(0.40, 2.40, 0.01)
    y = np.arange(0.40, 2.40, 0.01)
    x_values = np.array((0.49,0.56,0.67,0.84,1.66,2.22))
    fig, ax = plt.subplots()

    # ax.plot(x, y, color='black')

    ax.axvspan(0.45, 0.52, alpha=0.2, color='blue')
    ax.axvspan(0.52, 0.60, alpha=0.2, color='green')
    ax.axvspan(0.63, 0.69, alpha=0.2, color='red')
    ax.axvspan(0.77, 0.90, alpha=0.2, color='grey')
    ax.axvspan(1.55, 1.75, alpha=0.2, color='grey')
    ax.axvspan(2.08, 2.35, alpha=0.2, color='grey')

    plt.xlabel("Wavelength μm")
    plt.ylabel("Normalized Intensity")

    clean_value = []
    debris_value = []
    background_value = []

    for image_path in images:
        _filename = image_path.split("/")[-1].split(".")[0]
        raster_img = rasterio.open(image_path)
        img_np = np.moveaxis(raster_img.read(), 0, 2)
        img_np[np.isnan(img_np)] = 0
        vector_crs = rasterio.crs.CRS(clean.crs)
        if vector_crs != raster_img.meta["crs"]:
            clean = clean.to_crs(raster_img.meta["crs"].data)
            debris = debris.to_crs(raster_img.meta["crs"].data)
        clean_mask = rasterio_mask(raster_img, list(clean.geometry), crop=False)[0]
        clean_mask = clean_mask[0, :, :]
        debris_mask = rasterio_mask(raster_img, list(debris.geometry), crop=False)[0]
        debris_mask = debris_mask[0, :, :]

        clean_index = np.argwhere(clean_mask != 0)
        debris_index = np.argwhere(debris_mask != 0)
        background_index = np.argwhere((clean_mask+debris_mask) == 0)

        np.random.shuffle(clean_index)
        np.random.shuffle(debris_index)
        np.random.shuffle(background_index)

        clean_index = clean_index[:n_points].tolist()
        debris_index = debris_index[:n_points].tolist()
        background_index = background_index[:n_points].tolist()
        print("Shuffle complete")

        i = 0
        print("Working on clean glaciers")
        for index in clean_index:
            if i >= max_points_per_label:
                break
            np_clean = img_np[index[0],index[1],:]
            if check_condition(np_clean):
                clean_value.append(np_clean)
                to_append = list(np_clean)
                to_append.append("clean")
                a_series = pd.Series(to_append, index=df.columns)
                df = df.append(a_series, ignore_index=True)
                i+=1

        i = 0
        print("Working on debris glaciers")
        for index in debris_index:
            if i >= max_points_per_label:
                break
            np_debris = img_np[index[0],index[1],:]
            if check_condition(np_debris):
                debris_value.append(np_debris)
                to_append = list(np_debris)
                to_append.append("debris")
                a_series = pd.Series(to_append, index=df.columns)
                df = df.append(a_series, ignore_index=True)
                i+=1

        i = 0
        print("Working on background")
        for index in background_index:
            if i >= max_points_per_label:
                break
            np_background = img_np[index[0],index[1],:]
            if check_condition(np_background):
                background_value.append(np_background)
                to_append = list(np_background)
                to_append.append("background")
                a_series = pd.Series(to_append, index=df.columns)
                df = df.append(a_series, ignore_index=True)
                i+=1

        clean_value_np = np.asarray(clean_value)
        clean_mean = clean_value_np.mean(axis=0)
        clean_std = clean_value_np.std(axis=0)
        clean_mean = np.append(clean_mean[0:5],clean_mean[5])
        # clean_mean = (clean_mean-norm_mean)/norm_std
        debris_value_np = np.asarray(debris_value)
        debris_mean = debris_value_np.mean(axis=0)
        debris_std = debris_value_np.std(axis=0)
        debris_mean = np.append(debris_mean[0:5],debris_mean[5])
        # debris_mean = (debris_mean-norm_mean)/norm_std
        background_value_np = np.asarray(background_value)
        background_mean = background_value_np.mean(axis=0)
        background_std = background_value_np.std(axis=0)
        background_mean = np.append(background_mean[0:5],background_mean[5])
        # background_mean = (background_mean-norm_mean)/norm_std
              
        for (x,y) in zip(x_values,clean_mean):
            ax.plot(x, y, 'bo')

        for (x,y) in zip(x_values,debris_mean):
            ax.plot(x, y, 'ro')

        for (x,y) in zip(x_values,background_mean):
            ax.plot(x, y, 'go')

        plt.plot(x_values, clean_mean, color='blue', label="Clean Glaciers")
        plt.plot(x_values, debris_mean, color='red', label="Debris Glaciers")
        plt.plot(x_values, background_mean, color='green', label="Background")
        plt.title("Wavelength vs Normalized intensity")
        ax.legend(bbox_to_anchor=(0.65, 1), loc='upper left', borderaxespad=0.)

plt.savefig(output_filename)
df.to_csv("data_"+name_pattern+".csv", index=False)