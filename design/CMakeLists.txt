cmake_minimum_required(VERSION 3.18)
project(mirp_sim LANGUAGES CXX CUDA)

# Enable C++17 (or newer)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED)
message(STATUS "MPI found: ${MPI_CXX_COMPILER}")

# Find CUDA (optional, but recommended)
find_package(CUDA QUIET)
if(CUDA_FOUND)
    message(STATUS "CUDA found: ${CUDA_VERSION}")
    enable_language(CUDA)
    set(USE_CUDA TRUE)
else()
    message(STATUS "CUDA NOT found - building CPU only")
    set(USE_CUDA FALSE)
endif()

# Enable OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(WARNING "OpenMP NOT found")
endif()

# MFEM (you need to set MFEM_DIR or MFEM_ROOT in environment)
find_package(MFEM REQUIRED)
message(STATUS "MFEM include dir: ${MFEM_INCLUDE_DIR}")
message(STATUS "MFEM libraries: ${MFEM_LIBRARIES}")

# Source files
set(SOURCES
    main.cpp
    config/cgns_interface.cpp
    config/physics_loader.cpp
    core/mirp_sim.cpp
    core/memory_manager.cpp
    setup/mesh_loader.cpp
    setup/boundary_conditions.cpp
    setup/field_initializer.cpp
    setup/domain_partition.cpp
    solve/cpu/elasticity_solver.cpp
    solve/gpu/elasticity_gpu.cpp
    solve/gpu/kernels/curvature_kernels.cu
    solve/gpu/kernels/traction_kernels.cu
    solve/mpi/mpi_solver_driver.cpp
    solve/solver_manager.cpp
    analysis/stress_analysis.cpp
    analysis/surface_analysis.cpp
    analysis/region_analysis.cpp
    analysis/result_fields.cpp
    output/results_writer.cpp
    output/logger.cpp
    output/timers.cpp
    utils/math_utils.cpp
    utils/device_utils.cpp
    utils/string_utils.cpp
)

# Include directories
include_directories(
    ${MFEM_INCLUDE_DIR}
    ${MPI_CXX_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Executable target
if(USE_CUDA)
    cuda_add_executable(mirp_sim ${SOURCES})
    target_compile_definitions(mirp_sim PRIVATE USE_CUDA=1)
else()
    add_executable(mirp_sim ${SOURCES})
endif()

# Link libraries
target_link_libraries(mirp_sim
    ${MFEM_LIBRARIES}
    MPI::MPI_CXX
)

# Link CUDA libraries if used
if(USE_CUDA)
    target_link_libraries(mirp_sim cuda cudart)
endif()

# OpenMP flags
if(OpenMP_CXX_FOUND)
    target_link_libraries(mirp_sim OpenMP::OpenMP_CXX)
endif()

# Set compiler warnings (optional but recommended)
target_compile_options(mirp_sim PRIVATE -Wall -Wextra -Wpedantic)
