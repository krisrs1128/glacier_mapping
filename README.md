# Setup and training mapping models

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
   ![storage](https://drive.google.com/uc?id=1wo7F_1mD4ueLbxOt_s5_Mc0s-YhCyBmD)
   and then add a `file share`,
   ![share](https://drive.google.com/uc?id=1bJ95Lg13FjXLvGWrwZm-0xbdNJQVfbb1)
2. `tar -zcvf` all the processed data on the compute canada cluster.
3. Log into the VM on which you want to work with the data.
4. On that VM, mount the file share, by executing the script provided in the
   connect button,
![connect](https://drive.google.com/uc?id=1tcOZFKqeW6UIOA2HHa1xbampIlx9s7Da)
5. `rsync` the zipped data onto the VM that you're on.`
6. `tar -zxvf` the data under the `/mnt/storagename` directory created by the
   script in step 4.

### Running  
Note: This project requires python 3.6 or higher.

Once you've executed the setup above, you can start a training run by navigating
to the `/home/kris/` directory and running,

```
sh azure.sh
cd glacier_mapping
python3 -m src.exper -c=shared/azure.yaml -o /directory/to/save/outputs
```

The `mount.sh` script is exactly the script from step 4 in the setup above. It's
needed because when you shutdown the machine, the mounting information gets
lost.

### Running on Azure using conda environment
```
conda create -n glacier_mapping python=3.6
conda activate glacier_mapping
cd glacier_mapping
pip install -r requirements.txt
python -m src.exper -c=shared/azure.yaml -o /directory/to/save/outputs
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

## pre-processing and slicing
To the run the preprocessing you need two folders, `img_data` and `vector_data`, which contains the tiffs and the labels respectively. The folders should be organized as illustrated bellow. 
```
data/
  img_data/ # tiff files
    2010/
      nepal/
      bhutan/
      ...
    2000/
      nepal/
      ...
  vector_data/ # labels, borders, and test labels
    2010/
      nepal/ # nepal labels
      ....
    2000/
      ....
    basin/
      data/ #.shp files for test labels
    borders/
      nepal/
        data/ #.shp files to nepal borders
      bhutan/
      ....
```
To run the pre-processing script
`python -m src.slice_n_preprocess -c conf/preprocessing_conf.yaml`
where `preprocessing_conf.yaml` is the configuration file for which years/countries to process and how to split and filter train/test. You can either choose a subset of years/countries or process all available data.

Example:

`year: ['2010'] country:[]`
will process all countries in 2010.

You should end up with `slices` directory for all the sliced data and `sat_data.csv` for the metadata needed for training. The folder `sat_files` will be created for all intermediate results, for each set of year/country,  to help with future debugging and ease of incremental adding of data. This folder is of no importance for training.

The filtering of data depends, for now, on the amount of labels existing inside the countries border, amount of values out of the satellite image (nan), and amount of labels in general in the slice. Any slice that returns `False` for any of the conditions in the config file, will be marked as invalid. Development split is generated at random, while test split can be random or according to the intersection with the test vector data (check config file for more details)  

## filtering pseudo labels
In order to fill debris_perc column in the metadata, you should run the following command with the metadata file 
`python -m src.filter_debris_by_perc --data_path /data/sat_data.csv` 

The command expects `slices` and `sat_data.csv` to be previously generated.

This is a necessary step to train on pseudo labels, and can be skipped with regular glaciers.

## normalization
To generate `normalization_data.pkl`, you should run the following command
`python -m src.normalize --base_dir ./data`

The command expects `slices` and `sat_data.csv` to be previously generated.

The normalization operates on all channels/snow_index/borders/elevation/slope, and the right normalization factors will be picked up correctly relative to your training configuration.


## vector data sources
Labels : [ICIMOD](http://www.icimod.org/)

[(2000, Nepal)](http://rds.icimod.org/Home/DataDetail?metadataId=9351&searchlist=True):

Polygons older/newer than 2 years from 2000 are filtered out. Original collection contains few polygons from 1990s

[(2000, Bhutan)](http://rds.icimod.org/Home/DataDetail?metadataId=9357&searchlist=True):

Used as it's

[(2010, Nepal)](http://rds.icimod.org/Home/DataDetail?metadataId=9348&searchlist=True):

Polygons older/newer than 2 years from 2010 are filtered out. Original collection is for 1980-2010

[(2010, Bhutan)](http://rds.icimod.org/Home/DataDetail?metadataId=9358&searchlist=True):

Used as it's

Borders:
[Natural Earth: Admin-0 details](http://www.naturalearthdata.com/downloads/10m-cultural-vectors/)

Test Data: 
Dudh Koshi sub basin (provided directly from ICIMOD)



~
