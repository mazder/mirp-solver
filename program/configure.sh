#!/bin/bash

project_path=$(pwd)

# Create a build directory if it doesn't exist
if [ ! -d "build" ]; then
    mkdir build
else
    rm -rf build
    mkdir build
fi

# Navigate to the build directory
cd build

# Run CMake to configure the project
cmake -DCMAKE_BUILD_TYPE=Debug -DPROJECT_PATH="${project_path}" .. -Wno-dev

# If CMake was successful, output success message
if [ $? -eq 0 ]; then
    echo "CMake configuration successful!"
else
    echo "CMake configuration failed!"
    exit 1
fi

make -j$(nproc)

cd ..

#gdb ./build/rnd

#cgnscheck ./data/setup.cgns


#if [ -d "build" ]; then
#    rm -rf build
#fi
