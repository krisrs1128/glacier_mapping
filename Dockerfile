FROM ubuntu:18.04

RUN apt-get update
RUN apt-get upgrade
RUN apt-get install -y software-properties-common git vim
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt -y install build-essential
RUN apt-get install -y python3-pip python3-dev
RUN apt-get install -y gdal-bin python3-gdal

RUN git clone https://github.com/krisrs1128/glacier_mapping.git
WORKDIR glacier_mapping
RUN pip3 install -r requirements.txt