cmake_minimum_required(VERSION 3.10)
project(TestCuda)

# Set the CUDA toolkit directory (if needed, you can explicitly set it, e.g., /usr/local/cuda)
find_package(CUDA REQUIRED)

# Add the source file
set(SOURCES test_cuda.cu)

# Create the executable
cuda_add_executable(test_cuda ${SOURCES})

# Set the compiler flags
set_target_properties(test_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Optionally, set the CUDA architecture (e.g., for specific GPUs)
# set(CUDA_ARCHITECTURES 61)  # For Pascal architecture, adjust as needed

# Link libraries if necessary (e.g., cuDNN, etc.)
# target_link_libraries(test_cuda ${CUDA_LIBRARIES})
