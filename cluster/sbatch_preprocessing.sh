#!/bin/bash
#SBATCH --account=rpp-bengioy            # Yoshua pays for your job
#SBATCH --cpus-per-task=6                # Ask for 6 CPUs
#SBATCH --gres=gpu:1                     # Ask for 1 GPU
#SBATCH --mem=32G                        # Ask for 32 GB of RAM
#SBATCH --time=5:00:00                   # The job will run for 5 hours
#SBATCH -o /scratch/shimaa/logs/glaciers_preprocessing-%j.out  # Write the log in $SCRATCH
#SBATCH -e /scratch/shimaa/logs/glaciers_preprocessing-%j.err  # Write the err in $SCRATCH

# change the log directories based on your space
# TODO: make the logging directories map to current user space

# Copy and unzip the raw data to the compute node
# this assumes you have the raw_data on your scratch space,
# if it doesn't, you can copy it from /scratch/shimaa (everyone has read access)
cp $SCRATCH/data/raw_glaciers_data.zip $SLURM_TMPDIR
unzip $SLURM_TMPDIR/raw_glaciers_data.zip -d $SLURM_TMPDIR

module load singularity/3.4
cd $HOME/glacier_mapping
# TODO: put all in one command
singularity exec --bind $SLURM_TMPDIR /scratch/sankarak/images/glaciers.sif python3 -W ignore -m src.slice_n_preprocess -c conf/preprocessing_conf.yaml
singularity exec --bind $SLURM_TMPDIR /scratch/sankarak/images/glaciers.sif python3 -W ignore -m src.filter_debris_by_perc --data_path $SLURM_TMPDIR/data/sat_data.csv
singularity exec --bind $SLURM_TMPDIR /scratch/sankarak/images/glaciers.sif python3 -m src.normalize --base_dir $SLURM_TMPDIR/data
singularity exec --bind $SLURM_TMPDIR /scratch/sankarak/images/glaciers.sif python3 -m src.normalize --base_dir $SLURM_TMPDIR/data --data_file sat_data_p_deb.csv --normalization_data norm_p_deb.pkl

# tar (compressing takes a huge time) and copy the processed data on $SCRATCH
cd $SLURM_TMPDIR
tar -cf processed_glacier_data.tar data
cp $SLURM_TMPDIR/processed_glacier_data.tar $SCRATCH/data
chmod a+r $SCRATCH/data/processed_glacier_data.tar