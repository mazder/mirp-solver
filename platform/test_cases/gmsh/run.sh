#!/bin/bash

g++ -std=c++17 -I/usr/local/include/gmsh gmsh_test.cpp -L/usr/local/lib -lgmsh -o gmsh_test

./gmsh_test
