# Mapping setup

## Azure Cluster

### Setup

These steps are to setup a new Ubuntu machine from scratch. You can skip this
section if you're logging into our existing machine -- it already has all the
necessary packages and data.

To install all packages, run the `cluster/azure.sh` script.

For data, we've transferred all the processed numpy arrays into an `file
store`. The fastest way to do this ended up being 

1. Create a file share, using `portal.azure.com`. You need to first create a
   storage account,
   ![storage](https://drive.google.com/open?id=1wo7F_1mD4ueLbxOt_s5_Mc0s-YhCyBmD)
   and then add a `file share`,
   https://drive.google.com/open?id=1bJ95Lg13FjXLvGWrwZm-0xbdNJQVfbb1
   ![share](https://drive.google.com/open?id=1bJ95Lg13FjXLvGWrwZm-0xbdNJQVfbb1).
2. `tar -zcvf` all the processed data on the compute canada cluster.
3. Log into the VM on which you want to work with the data.
4. On that VM, mount the file share, by executing the script provided in the
   connect button,
![connect](https://drive.google.com/file/d/1tcOZFKqeW6UIOA2HHa1xbampIlx9s7Da/view?usp=sharing)
5. `rsync` the zipped data onto the VM that you're on.`
6. `tar -zxvf` the data under the `/mnt/storagename` directory created by the
   script in step 4.

### Running

Once you've executed the setup above, you can start a training run by navigating
to the `/home/kris/glacier_mapping` directory and running,

```
python3 -m src.exper -c=shared/azure.yaml -o /directory/to/save/outputs
```

## Compute Canada

### Environment

To load all the necessary software, you can load the [singularity
image](https://drive.google.com/open?id=1Dbd1Wae_Jf6BdhV2LkMaGjO8MwK5Lw4r) like
so,

```
module load singularity/3.4; singularity shell --nv --bind /scratch/sankarak/data/glaciers,/scratch/sankarak/glaciers /scratch/sankarak/images/glaciers.sif
```

## parallel training runs

To run many experiments in parallel, you can use the parallel run command. 

```
python3 parallel_run.py -e conf/explore-lr.yaml
```

The different jobs are specified by the different headings in the exploration
file (`-e`). You can see the example
[here](https://github.com/Sh-imaa/glacier_mapping/blob/master/conf/explore-lr.yaml).
Anything not specified will go to the
[defaults](https://github.com/Sh-imaa/glacier_mapping/blob/master/shared/defaults.yaml).

## single training run

You can point the `exper` script to a single training configuration file,

```
python3 -m src.exper -c /scratch/sankarak/glaciers/explore-lr-experiment--16/run_1/random-rates.yaml -o .
```

for example. The `-c` tells you the path to the config, and `-o` says where to write the output (checkpoints, logged metrics).

## looking at results

The results get saved to the output directory. You can use `wandb sync` to upload the results to some [pretty viewers](https://app.wandb.ai/krisrs1128/glacier_mapping/runs/rsii7qj6?workspace=default).
~
