#!/usr/bin/env bash
#
# Azure setup =(almost) Singularity def
#
sudo apt-get update
sudo apt -y upgrade

sudo apt -y install software-properties-common
sudo apt -y install unzip
sudo apt -y install python3-pip
sudo apt -y install build-essential libssl-dev libffi-dev python3-dev
sudo apt -y install nano vim ipython3

pip3 install numpy pandas wandb Pillow addict pyyaml rasterio
pip3 install --no-cache-dir torch torchvision matplotlib seaborn
