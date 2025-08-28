# Specify the minimum version of CMake required
cmake_minimum_required(VERSION 3.10)

# Project name
project(MPIExample)

# Find the MPI package
find_package(MPI REQUIRED)

# Add the executable and source files
add_executable(mpi_program main.cpp)

# Link MPI libraries to your executable
target_link_libraries(mpi_program MPI::MPI_CXX)

# Optionally, set additional compile flags if needed
target_compile_options(mpi_program PRIVATE -O2)

# If you need to use CUDA, OpenMP, or other libraries, add them here
# Example:
# target_link_libraries(mpi_program CUDA::cudart)
