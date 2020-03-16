#!/usr/bin/env python
from pathlib import Path
import numpy as np
import pandas as pd
import os

def slice_sbatch(metadata_path, job_dir=None, log_dir=None, n_jobs=10):

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
            fh.writelines(f"#SBATCH --job-name=slice_{i}.job\n")
            fh.writelines(f"#SBATCH --output={log_dir}/slice_{i}.out\n")
            fh.writelines(f"#SBATCH --error={log_dir}/slice_{i}.err\n")
            fh.writelines("#SBATCH --time=00:05:00\n")
            fh.writelines("#SBATCH --mem=4G\n")
            fh.writelines("#SBATCH --cpus-per-task=1\n")
            fh.writelines("#SBATCH --qos=normal\n")
            fh.writelines("#SBATCH --mail-type=ALL\n")
            fh.writelines("#SBATCH --mail-user=sankaran.kris@gmail.com\n")
            fh.writelines("cd $HOME/glacier_mapping")
            fh.writelines("module load singularity/3.5")
            fh.writelines("cd $HOME/glacier_mapping")
            fh.writelines(f"singularity exec --bind $SCRATCH/data/ $SCRATCH/images/glaciers.sif python3 -m src.slice -p {metadata_path} -s {split_points[i]} -e {split_points[i + 1]} -c 5")

        os.system(f"sbatch {str(job_file)}")


if __name__ == "__main__":
    default_root = Path(os.environ["SCRATCH"], "data", "glaciers")
    metadata_path = Path(default_root, "masks", "metadata.csv")
    slice_sbatch(metadata_path, Path(os.environ["HOME"], "jobs"), Path(os.environ["HOME"], "logs"))
