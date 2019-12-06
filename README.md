# glacier_mapping 

_This only works for the Mila cluster... need to setup for Azure._

## environment

To load all the necessary software, you can load the [singularity image](https://drive.google.com/open?id=1Dbd1Wae_Jf6BdhV2LkMaGjO8MwK5Lw4r) like so, 

```
module load singularity/3.4; singularity shell --nv --bind /scratch/sankarak/data/glaciers,/scratch/sankarak/glaciers /scratch/sankarak/images/glaciers.sif
```

We'll just replace the [recipe](https://github.com/Sh-imaa/glacier_mapping/blob/master/cluster/glaciers.def) with a bash script in the azure (eventually...).

## parallel training runs

To run many experiments in parallel, you can use the parallel run command. 

```
python3 parallel_run.py -e conf/explore-lr.yaml
```

The different jobs are specified by the different headings in the exploration file (`-e`). You can see the example [here](https://github.com/Sh-imaa/glacier_mapping/blob/master/conf/explore-lr.yaml). Anything not specified will go to the [defaults](https://github.com/Sh-imaa/glacier_mapping/blob/master/shared/defaults.yaml).

## single training run

You can point the `exper` script to a single training configuration file,

```
python3 -m src.exper -c /scratch/sankarak/glaciers/explore-lr-experiment--16/run_1/random-rates.yaml -o .
```

for example. The `-c` tells you the path to the config, and `-o` says where to write the output (checkpoints, logged metrics). Here's what that yaml file looked like,
```
data:
  borders: true
  load_limit: -1
  metadata: sat_data.csv
  original_path: /scratch/sankarak/data/glaciers/
  path: /scratch/sankarak/data/glaciers/
model:
  inchannels: 11
  net_depth: 4
  outchannels: 1
train:
  batch_size: 4
  infer_every_steps: 5000
  lr:
    from:
    - 1.0e-05
    - 0.01
    sample: uniform
  n_epochs: 100
  num_workers: 1
  save_freq: 2
  shuffle: true
  store_images: false
```

## looking at results

The results get saved to the output directory. You can use `wandb sync` to upload the results to some [pretty viewers](https://app.wandb.ai/krisrs1128/glacier_mapping/runs/rsii7qj6?workspace=default).
~
