cmake_minimum_required(VERSION 3.18)
project(cuda_matrix_ops LANGUAGES CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Tell CMake to build a shared library
add_library(cuda_matrix_ops SHARED matrix_ops.cu)

set_target_properties(cuda_matrix_ops PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Allow unsupported host compilers (e.g., GCC 14)
target_compile_options(cuda_matrix_ops PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--allow-unsupported-compiler;-Xcompiler=-w>
)

# Optional headers
target_include_directories(cuda_matrix_ops PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
