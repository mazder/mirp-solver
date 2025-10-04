#!/bin/bash

# === 2. Setup Environment ===
export CC=/opt/gcc-14/bin/gcc
export CXX=/opt/gcc-14/bin/g++
export PATH=/opt/mpich/bin:/opt/gcc-14/bin:$PATH
export LD_LIBRARY_PATH=/opt/mpich/lib:/opt/gcc-14/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=/opt/mpich/lib:/opt/gcc-14/lib64:$LIBRARY_PATH

export LD_LIBRARY_PATH=/opt/hypre/lib:$LD_LIBRARY_PATH
export PATH=/opt/hypre/bin:/opt/hypre/include:$PATH

export LD_LIBRARY_PATH=/opt/gklib/lib:$LD_LIBRARY_PATH
export PATH=/opt/gklib/bin:/opt/gklib/include:$PATH

export LD_LIBRARY_PATH=/opt/metis5/lib:$LD_LIBRARY_PATH
export PATH=/opt/metis/bin:/opt/metis5/include:$PATH


/opt/mpich/bin/mpicxx ex1.cpp \
    -I/opt/mfem/include \
    -I/opt/hypre/include \
    -I/opt/metis/include \
    -I/opt/gklib/include \
    -I/usr/local/cuda/include \
    -L/usr/local/lib -lmfem \
    -L/opt/hypre/lib -lHYPRE \
    -L/opt/metis5/lib -lmetis \
    -L/opt/gklib/lib -lgklib \
    -L/opt/mpich/lib -lmpi \
    -L/usr/local/cuda/lib64 -lcudart -lcublas -lcusparse \
    -o ex1 \
    -std=c++17 -O3 -fopenmp

ldd ./ex1
mpirun -np 4 ./ex1 -m ./star.mesh
