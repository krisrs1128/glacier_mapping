import numpy as np
import os
import glob
import json

imgs = glob.glob('/scratch/akera/glacier_slices/*img*')

def generate_stats(image_paths, sample_size, outpath="stats.json"):
    """
    Function to generate statistics of the input image channels

    :param image_paths: List of Paths to images in directory
    :param sample_size: int giving the size of the sample from which to compute the statistics
    :param outpath: str The path to the output json file containing computed statistics

    :return Dictionary with keys for means and stds across the channels in input images
    """
    image_paths = np.random.choice(image_paths, sample_size, replace=False)
    images = [np.load(image_path) for image_path in image_paths]
    batch = np.stack(images)
    means = np.nanmean(batch, axis=(0,1,2))
    stds = np.nanstd(batch, axis=(0,1,2))

    with open(outpath, "w") as f:
        stats = {
            "means": means.tolist(),
            "stds": stds.tolist()
        }

        json.dump(stats,f)

    return(stats)


def normalize(image, means, stds):
    """
        :param image: Input image to normalize
        :param means: Computed mean of the input channels
        :param stds: Computed standard deviation of the input channels

        :return image: Normalized image
    """
    for i in range(image.shape[2]):
        image[:,:,i] -= means[i]
        if stds[i]>0:
            image[:,:,i] /= stds[i]
        else:
            image[:,:,i] = 0

    return (image)



if __name__ == "__main__":
    stats = generate_stats(imgs, 5)
    image = np.load(imgs[1]) 
    image_ = normalize(image, **stats)
