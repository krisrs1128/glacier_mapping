#!/usr/bin/env bash
#
# Azure setup =(almost) Singularity def
#

curl https://packages.microsoft.com/config/ubuntu/18.04/prod.list > ./microsoft-prod.list
sudo cp ./microsoft-prod.list /etc/apt/sources.list.d/
curl https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > microsoft.gpg
sudo cp ./microsoft.gpg /etc/apt/trusted.gpg.d/
sudo apt-get update
sudo apt -y upgrade

sudo apt-get install blobfuse
sudo apt -y install software-properties-common
sudo apt -y install unzip
sudo apt -y install blobfuse
sudo apt -y install python3-pip
sudo apt -y install build-essential libssl-dev libffi-dev python3-dev
sudo apt -y install nano vim

pip3 install numpy pandas wandb Pillow addict pyyaml
pip3 install --no-cache-dir torch torchvision matplotlib ipython seaborn


blobfuse ~/data -- tmp-path=/mnt/resource/blobfusetmp — config-file=./fuse_connection.cfg -o attr_timeout=240 -o entry_timeout=240 -o negative_timeout=120 -o allow_other
