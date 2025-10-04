#!/bin/bash
rm -rf build
mkdir build && cd build
cmake .. -DMFEM_DIR=/usr/local/lib/cmake/mfem
make -j$(nproc)


mpirun -np 1 ./main -m ../output/tet_mesh.msh -d cpu
