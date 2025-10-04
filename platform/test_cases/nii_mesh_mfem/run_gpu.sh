#!/bin/bash
rm -rf build
mkdir build && cd build
cmake .. -DMFEM_DIR=/usr/local/lib/cmake/mfem
make -j$(nproc)
mpirun -np 4 ./main -m ../tumor_mesh_3d.msh --device cuda

cd ..
/opt/gmsh/bin/gmsh tumor_mesh_3d.msh -0 -format cgns -o tumor_mesh.cgns
