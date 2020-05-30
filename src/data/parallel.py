#!/usr/bin/env python
from pathlib import Path
import numpy as np
import pandas as pd
import os
import glob
import geopandas as gpd
import argparse

def slice_sbatch(metadata_path, job_dir=None, log_dir=None, n_jobs=10):
    """
    Write and Launch sbatch Jobs for Slicing

    slice.py lets us slice any collection of tiff / mask pairs, but goes
    through the images in order. If a slurm cluster is available, this function
    will parallelize that operation, by writing separate sbatch job scripts to
    the job_dir directory, and then launching them on the associated slurm
    cluster.

    :param metadata_path: The path to the metadata file giving information for
      all of the masks.
    :param job_dir: The directory to write the sbatch job scripts that will be
      launched.
    :param log_dir: The directory to write the logs from the slurm jobs.
    :param n_jobs: The number of slurm jobs to launch (the degree of parallelization)
    :returns None: Returns nothing, but writes slice*.geojson and slices in the
      default output directory of slice.py
    """
    # prepare output directories
    if not job_dir:
        job_dir = os.getcwd()
    if not log_dir:
        log_dir = os.getcwd()
    os.makedirs(job_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # find which tiffs each job will be slicing
    metadata = pd.read_csv(metadata_path)
    n_rows = len(metadata)
    split_points = np.arange(0, n_rows, n_rows // n_jobs)
    if split_points[-1] != n_rows:
        split_points = np.append(split_points, n_rows)

    for i in range(len(split_points) - 1):

        job_file = Path(job_dir, f"slice_{i}.job")
        with open(job_file, "w") as fh:
            fh.writelines("#!/bin/bash\n")
            fh.writelines(f"#SBATCH --job-name={job_dir}/slice_{i}.job\n")
            fh.writelines(f"#SBATCH --output={log_dir}/slice_{i}_%j.out\n")
            fh.writelines(f"#SBATCH --error={log_dir}/slice_{i}_%j.err\n")
            fh.writelines("#SBATCH --time=00:06:30\n")
            fh.writelines("#SBATCH --mem=32G\n")
            fh.writelines("#SBATCH --cpus-per-task=2\n")
            fh.writelines("#SBATCH --qos=normal\n")
            fh.writelines("#SBATCH --mail-type=ALL\n")
            fh.writelines("#SBATCH --mail-user=sankaran.kris@gmail.com\n")
            fh.writelines("cd $HOME/glacier_mapping\n")
            fh.writelines("module load singularity/3.5\n")
            fh.writelines("cd $HOME/glacier_mapping\n")
            fh.writelines(f"singularity exec --bind $SCRATCH/data/ $SCRATCH/images/glaciers.sif python3 -m src.slice -p {metadata_path} -s {split_points[i]} -e {split_points[i + 1]} -c 5 -b slice_{i}")

        os.system(f"sbatch {str(job_file)}")

def merge_geojson(source_paths, output_path):
    sources = [gpd.read_file(s) for s in source_paths]
    #TODO: looks like gpd has no concatenate. (Anthony, Bibek)
    output = gpd.concatenate(sources)
    output.to_file(output_path, driver="GeoJSON")


if __name__ == "__main__":
    default_root = Path(os.environ["SCRATCH"], "data", "glaciers")
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--type", type=str, default="slice")
    args = parser.parse_args()


    if args.type == "slice":
        metadata_path = Path(default_root, "masks", "metadata.csv")
        slice_sbatch(metadata_path, Path(os.environ["HOME"], "jobs"), Path(os.environ["HOME"], "logs"))
    elif args.type == "merge":
        source_paths = glob.glob(Path(default_root, "slices", "*.geojson"))
        output_path = glob.glob(Path(default_root, "slices", "slices_metadata.geojson"))
        merge_geojson(source_paths, output_path)
    else:
        pass
