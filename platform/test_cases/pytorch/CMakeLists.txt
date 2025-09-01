cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(MyLibTorchProject)

# Specify the path to the LibTorch installation
set(TORCH_PATH "/opt/libtorch")

# Set C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

# Specify CUDA and cuDNN paths
set(CUDNN_INCLUDE_DIR "/usr/local/cuda/include")
set(CUDNN_LIB_DIR "/usr/local/cuda/lib64")
set(CUDA_PATH "/usr/local/cuda")

set(USE_CUDNN 1)

# Add CUDA and cuDNN include directories
include_directories(
    ${CUDNN_INCLUDE_DIR}
    ${TORCH_PATH}/include
    ${TORCH_PATH}/include/torch/csrc/api/include
    ${TORCH_PATH}/include/torch/csrc/autograd
)

# Set the path to the Torch CMake directory
set(Torch_DIR "${TORCH_PATH}/share/cmake/Torch")

# Find the Torch package
find_package(Torch REQUIRED)

# Find the CUDA package (specify version)
find_package(CUDA REQUIRED)

# Link directories for CUDA and cuDNN
link_directories(
    ${CUDNN_LIB_DIR}
    ${TORCH_PATH}/lib
    ${CUDA_PATH}/lib64  # Include CUDA lib64 path for linking
)

# Enable CUDA language support
enable_language(CUDA)

# Add the executable
add_executable(main_app main.cpp)

# Link libraries for LibTorch, CUDA, cuDNN, and NVTX
target_link_libraries(main_app PRIVATE
    ${TORCH_LIBRARIES}
    ${CUDA_LIBRARIES}
    cudnn
)
