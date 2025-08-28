# CMakeLists.txt for CUDA + cuDNN project

cmake_minimum_required(VERSION 3.10)
project(test_cudnn)

# Specify the CUDA version
find_package(CUDA REQUIRED)

# Specify the paths for cuDNN (if needed)
set(CUDNN_INCLUDE_DIR /usr/local/cuda/include)
set(CUDNN_LIB_DIR /usr/local/cuda/lib64)

# Include the CUDA and cuDNN directories
include_directories(${CUDNN_INCLUDE_DIR})
link_directories(${CUDNN_LIB_DIR})

# Define the source file
set(SOURCE_FILES test_cudnn.cpp)

# Define the executable
add_executable(test_cudnn ${SOURCE_FILES})

# Link the libraries
target_link_libraries(test_cudnn ${CUDA_LIBRARIES} cudnn)
