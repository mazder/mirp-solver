#!/bin/bash
# ***************************************************************
# File Name    : docker_build.sh
# Author       : MD MAZDER RAHMAN
# Date         : August 27, 2025
# Description  : This script is a part of developing mirp project
# Version      : 1.0
# License      : All rights reserved
# ***************************************************************

# Check if GPUs are available (optional)
if ! nvidia-smi; then
  echo "No GPU available. Exiting."
  exit 1
fi

# Build the Docker image
docker build -t mirp/covptu-3dr-light-py39-light2:latest .

#xhost +local:root
# Run the Docker container with GPU support
# sudo docker run --gpus all -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --rm -v /home/mazder/mirp-solver:/workspace/mirp-solver mirp/covptu-3dr-light-py39-light2:latest

docker images -f "dangling=true"
docker image prune -f
