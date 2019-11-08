#!/usr/bin/env bash
#
# Azure setup =(almost) Singularity def
#

apt -y update
apt -y upgrade
apt -y install software-properties-common
apt -y install python3-pip
apt -y install build-essential libssl-dev libffi-dev python3-dev
apt -y install nano vim

pip3 install numpy pandas wandb Pillow addict pyyaml
pip3 install --no-cache-dir torch torchvision matplotlib ipython seaborn
