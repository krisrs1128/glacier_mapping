#!/usr/bin/env bash
FROM ubuntu:18.04

RUN apt-get update
RUN apt -y upgrade

RUN apt -y install software-properties-common
RUN apt -y install python3-pip
RUN apt -y install build-essential libssl-dev libffi-dev python3-dev
RUN apt -y install vim ipython3 git

RUN add-apt-repository ppa:ubuntugis/ubuntugis-unstable
RUN apt-get update
RUN apt -y install gdal-bin libgdal-dev
RUN pip3 install --upgrade pip
RUN pip3 install git+https://github.com/krisrs1128/glacier_mapping.git
