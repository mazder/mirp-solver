cmake_minimum_required(VERSION 3.16)
project(main LANGUAGES CXX)

# Find MFEM
find_package(MFEM REQUIRED HINTS /usr/local /opt/mfem /usr/local/lib/cmake/mfem)

# Main executable
add_executable(main main.cpp)
target_link_libraries(main PRIVATE mfem)

# CUDA Toolkit (if needed)
find_package(CUDAToolkit REQUIRED)
target_link_libraries(main PRIVATE CUDA::cudart CUDA::cublas CUDA::cusparse)

# Include directories for external libs
target_include_directories(main PRIVATE
    /usr/local/cuda/include
    /opt/metis5/include
    /opt/gklib/include
)

# Link libraries
target_link_directories(main PRIVATE /opt/metis5/lib /opt/gklib/lib /opt/hypre/lib)
target_link_libraries(main PRIVATE metis gklib HYPRE)
