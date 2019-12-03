#!/bin/bash
#SBATCH --account=rpp-bengioy               # Yoshua pays for your job
#SBATCH --cpus-per-task=8       # Ask for 6 CPUs
#SBATCH --gres=gpu:1                        # Ask for 1 GPU
#SBATCH --mem=32G                 # Ask for 32 GB of RAM
#SBATCH --time=00:30:00
#SBATCH -o /home/bibek/logs/glaciers-%j.out  # Write the log in $SCRATCH

if [ -d "$SLURM_TMPDIR" ]; then
        cd /scratch/sankarak/data/glaciers
        zip -r glaciers.zip patches masks > /dev/null
    fi

    cp {zip_path} $SLURM_TMPDIR
    cd $SLURM_TMPDIR
    unzip {zip_name} > /dev/null

cd $HOME/glacier_mapping/

module load singularity/3.4
echo "Starting job"
singularity exec --nv --bind $SLURM_TMPDIR,/home/bibek/Desktop/Course_Materials/UTEPLabs/3rd Sem/glacier_mapping/explore-lr-experiment--1/run_0\
        /scratch/sankarak/images/glaciers.sif\
        python3 -m src.exper \
        -m "fixed rate" \
        -c "/home/bibek/Desktop/Course_Materials/UTEPLabs/3rd Sem/glacier_mapping/explore-lr-experiment--1/run_0/explore-lr.yaml"\
        -o "/home/bibek/Desktop/Course_Materials/UTEPLabs/3rd Sem/glacier_mapping/explore-lr-experiment--1/run_0"
