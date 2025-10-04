#!/bin/bash
rm -rf build
mkdir build && cd build
cmake .. -DMFEM_DIR=/usr/local/lib/cmake/mfem
make -j$(nproc)
mpirun -np 1 ./ex5 -m ../tumor_mesh_3d.msh --device cuda
