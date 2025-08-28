#!/bin/bash
# ***************************************************************
# File Name    : docker_engine.sh
# Author       : MD MAZDER RAHMAN
# Date         : January 11, 2025
# Description  : This script is a part of developing 3d-tumor-reconstruction
# Version      : 1.0
# License      : All rights reserved
# ***************************************************************

sudo apt-get update
sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common
    
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository \
    "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
    $(lsb_release -cs) \
    stable"
    
sudo apt-get update

sudo apt-get install docker-ce docker-ce-cli containerd.io

sudo docker --version


# Before build docker container, change the docker-data location
#  sudo nano /etc/docker/daemon.json
# {
#  "data-root": "/media/mazder/cf79910a-d4a6-4b65-8019-fad2ed770e03/home/mrahman/docker-data"
# }
# sudo systemctl restart docker
# sudo docker info | grep "Docker Root Dir"

