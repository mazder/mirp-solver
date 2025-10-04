cmake_minimum_required(VERSION 3.16)
project(ex5 LANGUAGES CXX)

# Find MFEM
find_package(MFEM REQUIRED HINTS /usr/local /usr/local/lib/cmake/mfem /opt/mfem)

add_executable(ex5 ex5.cpp)
target_link_libraries(ex5 PRIVATE mfem)

# CUDA runtime (for cudaLaunchKernel, cublas, cusparse)
find_package(CUDAToolkit REQUIRED)
target_link_libraries(ex5 PRIVATE CUDA::cudart CUDA::cublas CUDA::cusparse)

# METIS + GKlib (fix gk_* missing symbols)
target_link_directories(ex5 PRIVATE /opt/metis5/lib /opt/gklib/lib)
target_link_libraries(ex5 PRIVATE metis gklib)

# Include dirs
target_include_directories(ex5 PRIVATE
    /usr/local/cuda/include
    /opt/metis5/include
    /opt/gklib/include
)

