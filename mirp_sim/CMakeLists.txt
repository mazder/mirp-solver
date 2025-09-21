cmake_minimum_required(VERSION 3.18)
project(mirp_sim LANGUAGES CXX CUDA)

# ───────────────────────────────────────────────
# General Settings
# ───────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ───────────────────────────────────────────────
# CUDA Support (Modern CMake)
# ───────────────────────────────────────────────
option(USE_CUDA "Enable CUDA GPU acceleration" ON)

if(USE_CUDA)
    enable_language(CUDA)
    message(STATUS "✅ CUDA support enabled")
else()
    message(STATUS "⚠️ CUDA support disabled (CPU-only build)")
endif()

# ───────────────────────────────────────────────
# MPI Support
# ───────────────────────────────────────────────
find_package(MPI REQUIRED)
message(STATUS "MPI found: ${MPI_CXX_COMPILER}")

# ───────────────────────────────────────────────
# OpenMP Support
# ───────────────────────────────────────────────
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "✅ OpenMP support enabled")
else()
    message(WARNING "⚠️ OpenMP NOT found")
endif()

# ───────────────────────────────────────────────
# MFEM Support (Required)
# ───────────────────────────────────────────────
find_package(MFEM REQUIRED HINTS /usr/local /opt/mfem /usr/local/lib/cmake/mfem)

message(STATUS "MFEM include dir: ${MFEM_INCLUDE_DIR}")
message(STATUS "MFEM libraries: ${MFEM_LIBRARIES}")

# ───────────────────────────────────────────────
# Source Files
# ───────────────────────────────────────────────
set(SOURCES
    main.cpp

    # Config
    # config/cgns_interface.cpp
    # config/physics_loader.cpp

    # Core
    # core/mirp_sim.cpp
    # core/memory_manager.cpp

    # Setup
    # setup/mesh_loader.cpp
    # setup/boundary_conditions.cpp
    # setup/field_initializer.cpp
    # setup/domain_partition.cpp

    # CPU Solver
    # solve/cpu/elasticity_solver.cpp

    # GPU Solver
    # solve/gpu/elasticity_gpu.cpp
    # solve/gpu/kernels/curvature_kernels.cu
    # solve/gpu/kernels/traction_kernels.cu

    # MPI Solver
    # solve/mpi/mpi_solver_driver.cpp

    # Solver Control
    # solve/solver_manager.cpp

    # Analysis
    # analysis/stress_analysis.cpp
    # analysis/surface_analysis.cpp
    # analysis/region_analysis.cpp
    # analysis/result_fields.cpp

    # Output
    # output/results_writer.cpp
    # output/logger.cpp
    # output/timers.cpp

    # Utils
    # utils/math_utils.cpp
    # utils/device_utils.cpp
    # utils/string_utils.cpp
)

# ───────────────────────────────────────────────
# Include Directories
# ───────────────────────────────────────────────
include_directories(
    ${MFEM_INCLUDE_DIR}
    ${MPI_CXX_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# ───────────────────────────────────────────────
# Executable Target
# ───────────────────────────────────────────────
add_executable(mirp_sim ${SOURCES})

# Conditional definitions
if(USE_CUDA)
    target_compile_definitions(mirp_sim PRIVATE USE_CUDA=1)
endif()

# ───────────────────────────────────────────────
# Link Libraries
# ───────────────────────────────────────────────
target_link_libraries(mirp_sim
    ${MFEM_LIBRARIES}
    MPI::MPI_CXX
    mfem metis gklib HYPRE
)

# CUDA runtime libraries
if(USE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(mirp_sim
        CUDA::cudart
        CUDA::cublas
        CUDA::cusparse
    )
endif()

# OpenMP libraries
if(OpenMP_CXX_FOUND)
    target_link_libraries(mirp_sim OpenMP::OpenMP_CXX)
endif()

# ───────────────────────────────────────────────
# External Include/Link Directories (Optional)
# ───────────────────────────────────────────────
target_include_directories(mirp_sim PRIVATE
    /usr/local/cuda/include
    /opt/metis5/include
    /opt/gklib/include
)

target_link_directories(mirp_sim PRIVATE
    /opt/metis5/lib
    /opt/gklib/lib
    /opt/hypre/lib
)

# ───────────────────────────────────────────────
# Compiler Warnings
# ───────────────────────────────────────────────
target_compile_options(mirp_sim PRIVATE -Wall -Wextra -Wpedantic)
