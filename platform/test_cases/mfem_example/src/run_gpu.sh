#!/bin/bash
rm -rf build
mkdir build && cd build
cmake .. -DMFEM_DIR=/usr/local/lib/cmake/mfem
make -j$(nproc)
mpirun -np 4 ./ex1 -m ../star.mesh --device cuda
