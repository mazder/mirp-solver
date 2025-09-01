#!/bin/bash

set -e  # Exit on error

 #=== 1. Clean Previous Build ===
cd medsolver
rm -rf build
echo "[Clean] Removed previous build directory."

# === 2. Setup Environment ===
export CC=/opt/gcc-14/bin/gcc
export CXX=/opt/gcc-14/bin/g++
export PATH=/opt/mpich/bin:/opt/gcc-14/bin:$PATH
export LD_LIBRARY_PATH=/opt/mpich/lib:/opt/gcc-14/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=/opt/mpich/lib:/opt/gcc-14/lib64:$LIBRARY_PATH

export CUDAHOSTCXX=/usr/bin/g++

# Paths
export PATH=/opt/gcc-14/bin:/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/opt/gcc-14/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH


echo "[Env] GCC: $($CXX --version | head -n 1)"
echo "[Env] MPICH: $(mpicc --version | head -n 1)"

# === 3. Configure with CMake ===
mkdir -p build && cd build

cmake .. \
  -DVTK_DIR=/opt/vtk/lib/cmake/vtk-9.5 \
  -Dvtkbone_DIR=/opt/VTKbone/lib/cmake/vtkbone-2.0/vtkbone \
  -DCMAKE_BUILD_TYPE=Release

# === 4. Build ===
make -j$(nproc)
echo "[Build] medsolver built successfully."

# === 5. Sanity Test ===
#cd ../..
#./medsolver/build/bin/medsolver ./input/spineFE-data/nii_files test.vkt
#python3 src/calibration.py
