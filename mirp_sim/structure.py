import os

folders = [
    "config",
    "core",
    "setup",
    "solve/cpu",
    "solve/gpu/kernels",
    "solve/mpi",
    "analysis",
    "output",
    "utils",
    "tests",
    "scripts"
]

files = {
    "main.cpp": "// Main driver: initializes MPI, device, runs simulation\n",
    "config/cgns_interface.cpp": "// CGNS file reading/writing implementation\n",
    "config/cgns_interface.hpp": "// CGNS interface header\n",
    "config/physics_loader.cpp": "// Loads material properties\n",
    "config/physics_loader.hpp": "// Physics loader header\n",
    "core/mirp_sim.cpp": "// Main simulation controller implementation\n",
    "core/mirp_sim.hpp": "// MirpSim class header\n",
    "core/memory_manager.cpp": "// Memory manager implementation\n",
    "core/memory_manager.hpp": "// Memory manager header\n",
    "core/types.hpp": "// Common enums, constants\n",
    "setup/mesh_loader.cpp": "// Mesh loader implementation\n",
    "setup/mesh_loader.hpp": "// Mesh loader header\n",
    "setup/boundary_conditions.cpp": "// BC loading implementation\n",
    "setup/boundary_conditions.hpp": "// BC header\n",
    "setup/field_initializer.cpp": "// Initialize fields like displacement\n",
    "setup/field_initializer.hpp": "// Header for field init\n",
    "setup/domain_partition.cpp": "// MPI domain decomposition\n",
    "setup/domain_partition.hpp": "// Domain partition header\n",
    "solve/cpu/elasticity_solver.cpp": "// CPU-based elasticity solver\n",
    "solve/cpu/elasticity_solver.hpp": "// CPU solver header\n",
    "solve/gpu/elasticity_gpu.cpp": "// GPU-based elasticity solver (MFEM)\n",
    "solve/gpu/elasticity_gpu.hpp": "// GPU solver header\n",
    "solve/gpu/kernels/curvature_kernels.cu": "// CUDA kernels for curvature\n",
    "solve/gpu/kernels/traction_kernels.cu": "// CUDA kernels for traction\n",
    "solve/mpi/mpi_solver_driver.cpp": "// MPI coordination driver\n",
    "solve/mpi/mpi_solver_driver.hpp": "// MPI solver driver header\n",
    "solve/solver_manager.cpp": "// Solver manager chooses CPU/GPU/MPI\n",
    "solve/solver_manager.hpp": "// Solver manager header\n",
    "analysis/stress_analysis.cpp": "// Stress, strain, von Mises calculation\n",
    "analysis/stress_analysis.hpp": "// Stress analysis header\n",
    "analysis/surface_analysis.cpp": "// Surface curvature, traction calculations\n",
    "analysis/surface_analysis.hpp": "// Surface analysis header\n",
    "analysis/region_analysis.cpp": "// Stats by region\n",
    "analysis/region_analysis.hpp": "// Region analysis header\n",
    "analysis/result_fields.cpp": "// Prepare output fields\n",
    "analysis/result_fields.hpp": "// Result fields header\n",
    "output/results_writer.cpp": "// CGNS and VTU output writer\n",
    "output/results_writer.hpp": "// Output writer header\n",
    "output/logger.cpp": "// Logging (MPI rank aware)\n",
    "output/logger.hpp": "// Logger header\n",
    "output/timers.cpp": "// Timing utilities\n",
    "output/timers.hpp": "// Timer header\n",
    "utils/math_utils.cpp": "// Math helper functions\n",
    "utils/math_utils.hpp": "// Math utils header\n",
    "utils/device_utils.cpp": "// CUDA device utilities\n",
    "utils/device_utils.hpp": "// Device utils header\n",
    "utils/string_utils.cpp": "// String manipulation helpers\n",
    "utils/string_utils.hpp": "// String utils header\n",
    "tests/test_memory.cpp": "// Unit tests for memory manager\n",
    "scripts/convert_to_cgns.py": "#!/usr/bin/env python3\n# Converts mesh and mask to CGNS\n",
    "scripts/launch_slurm.sh": "#!/bin/bash\n# SLURM job launcher script\n"
}

def create_structure(base_path="."):
    for folder in folders:
        path = os.path.join(base_path, folder)
        os.makedirs(path, exist_ok=True)
        print(f"Created folder: {path}")
    for file_path, content in files.items():
        full_path = os.path.join(base_path, file_path)
        if not os.path.exists(full_path):
            with open(full_path, "w") as f:
                f.write(content)
            print(f"Created file: {full_path}")

if __name__ == "__main__":
    create_structure()
