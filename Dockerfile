FROM ubuntu:18.04

RUN apt-get update
RUN apt-get install -y software-properties-common git
RUN add-apt-repository ppa:ubuntugis/ppa
RUN apt-get update
RUN apt-get install -y gdal-bin
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y python3.8
RUN apt-get install -y libpython3.8-dev
RUN apt-get install -y python3-pip
RUN apt-get install -y python3-venv python3-virtualenv

RUN git clone https://github.com/krisrs1128/glacier_mapping.git
WORKDIR glacier_mapping
RUN pip3 install -r requirements.txt