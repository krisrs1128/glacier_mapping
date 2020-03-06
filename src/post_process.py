import pandas as pd
import numpy as np
import addict
from addict import Dict
import pdb
import glob
import os
from argparse import ArgumentParser


def filter_directory(input_dir, filter_perc=0.2, filter_channel=0):
    """
    Return Paths for Pairs passing Filter Criteria

    :param input_dir: The directory containing the slices output from slice.py.
    :param filter_perc: The minimum percentage 1's in the filter_channel needed
      to pass the filter.
    :param filter_channel: The channel to do the filtering on.
    """
    slices = pd.DataFrame(pathlib.Path(input_dir, "slices_metadata.csv"))
    keep_ids = []

    img_paths, mask_paths = slices["img_slice"].values, slices["mask_slices"].values
    for i, mask_path in enumerate(mask_paths):
        mask = np.load(mask_path)

        if i % 10 == 0:
            print(f"{i}/{len(img_paths)}")
        perc = mask[:, :, filter_channel].mean()

        if perc > filter_perc:
            keep_ids.append({
                "img": img_paths[i],
                "mask": mask_path
            })

    return keep_ids


def postprocess(img, mask, opts):
    """
    Function to postprocess a single image, mask pair.
    :return: Image and Mask within criteria of the filter_pairs func
    """
    if opts.filter:
        keep_image = filter_pairs(mask, opts.filter_channel, opts.filter_percentage)
        if not keep_image:
            return None, None
    return(img, mask)

def filter_pairs(mask, filter_channel, filter_percentage):
    """
    Function to filter a single mask.
      :param mask: Input label .npy 3 channel mask
      :param filter_channel: Channel in mask to be considered for filtering
      :param filter_percentage: Minimum Percentage of filter_channel which should be non-zero

      :return Boolean: True if mask passed is above minimum Percentage, False otherwise
    """
    percentage = mask[:,:,filter_channel].mean()
    if percentage > filter_percentage:
        return True
    return False

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-d", "--slice_dir", type=str, help="path to directory with all the slices")
    parser.add_argument("-o", "--output_dir", type=str, help="path to output directory for postprocessed files")
    args = parser.parse_args()

    opts = Dict({"filter": True, "filter_percentage": 0, "filter_channel":0})
    mask_paths = glob.glob(args.slice_dir +"*mask*")
    import pdb; pdb.set_trace()

    for path in mask_paths:
        mask = np.load(path)
        img_path = path.replace("mask","img")
        img = np.load(img_path)
        img,mask = postprocess(img,mask,opts)

        if img is not None:
            np.save(args.output_dir+os.path.basename(path), mask)
            np.save(args.output_dir+os.path.basename(img_path), img)
            print(args.output_dir)
