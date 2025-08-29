#!/bin/bash

project_path=$(pwd)

# Create a build directory if it doesn't exist
if [ -d "build" ]; then
    rm -rf build

fi
rm -rf ${project_path}/data/*
echo "Clean up successful!"
