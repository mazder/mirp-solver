#!/bin/bash

mkdir build
cd build

cmake ..
make

./main_app
