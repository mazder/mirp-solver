# MIRP_SIM: Multi-physics Imaging-based Realistic Physics Simulation

## Overview

MIRP_SIM is a modular, high-performance finite element simulation framework designed for bone and tumor mechanics simulations.
It supports CPU and GPU acceleration, MPI-based parallelism, and flexible physics setups via CGNS input files.

---

## Features

- **Linear elasticity solver** (bone and tumor materials)
- **Surface analysis:** curvature, traction, and gamma computations
- **MPI-based distributed parallelism**
- **Multi-GPU support using CUDA and MFEM**
- **Configurable physics and mesh via CGNS files**
- **Modular memory management for CPU and GPU**
- **Output in CGNS and VTU for postprocessing**
- **OpenMP parallelism for CPU solvers**

---

## Directory Structure


```mirp_sim/
├── config/ # CGNS I/O, physics parameter loading
├── core/ # Simulation controller, memory manager
├── setup/ # Mesh loading, BCs, partitioning
├── solve/ # CPU/GPU solvers and MPI driver
├── analysis/ # Postprocessing analyses
├── output/ # Output writers, logging, timers
├── utils/ # Helper utilities
├── tests/ # Unit tests
├── scripts/ # Python helpers and SLURM launcher
├── main.cpp # Entry point
└── CMakeLists.txt # Build system```

## Build Instructions

Requirements:

- C++17 compiler
- MPI library (OpenMPI or MPICH)
- CUDA Toolkit (optional for GPU)
- MFEM library installed and accessible
- CMake 3.18 or newer

## Build steps:

```bash
mkdir build && cd build
cmake .. -DUSE_CUDA=ON
make -j```

## Usage
srun/mpirun -n 4 ./mirp_sim path/to/input.cgns
