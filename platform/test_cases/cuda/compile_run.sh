#!/bin/bash

mkdir build
cd build

cmake .. -Wno-dev
make

./test_cuda
