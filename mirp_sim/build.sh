#!/bin/bash
#
# build.sh — Build script for MIRP_SIM
#
# Author:      MD Mazder Rahman
# Created:     September 21, 2025
# License:     Proprietary — All Rights Reserved
# Copyright © 2025 XXX Technologies
#
# Description:
#   Cleans, configures, and builds the MIRP_SIM finite element simulation framework.
#   Supports optional CUDA and MPI acceleration.
#

set -euo pipefail
IFS=$'\n\t'

# ───────────────────────────────────────────────
# 0. User Configuration
# ───────────────────────────────────────────────

USE_CUDA=ON                   # Enable CUDA support (ON/OFF)
MPI_PATH=/opt/mpich           # Path to MPI installation

# ───────────────────────────────────────────────
# 1. Clean Previous Build
# ───────────────────────────────────────────────

echo "[CLEAN] Removing previous build directory..."
rm -rf build
mkdir -p build

# ───────────────────────────────────────────────
# 2. Set Environment
# ───────────────────────────────────────────────

echo "[ENV] Setting up environment variables..."

export PATH="$MPI_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$MPI_PATH/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

echo "[ENV] C++ Compiler : $(${CXX:-g++} --version | head -n1)"
echo "[ENV] MPI Compiler : $(mpicc --version | head -n1)"
echo "[ENV] CUDA Version : $(nvcc --version | grep release || echo 'CUDA not found')"

# ───────────────────────────────────────────────
# 3. Configure with CMake
# ───────────────────────────────────────────────

echo "[CMAKE] Configuring project (USE_CUDA=$USE_CUDA)..."
cd build
cmake .. -DUSE_CUDA=$USE_CUDA

# ───────────────────────────────────────────────
# 4. Build Project
# ───────────────────────────────────────────────

echo "[BUILD] Building MIRP_SIM..."
make -j"$(nproc)"

echo "[SUCCESS] MIRP_SIM built successfully."

# ───────────────────────────────────────────────
# 5. Optional: Run Sanity Test
# ───────────────────────────────────────────────

# echo "[TEST] Running sanity check..."
# ./mirp_sim input.cgns
