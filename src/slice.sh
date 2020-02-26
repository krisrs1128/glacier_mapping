#!/usr/bin/env bash

for k in {0..9}; do
    sbatch slice.sbatch /scratch/sankarak/data/tmp_masks/paths.csv ${k}
done
