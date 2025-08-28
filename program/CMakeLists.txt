cmake_minimum_required(VERSION 3.10)
project(run_mirp)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find CUDA package
find_package(CUDA REQUIRED)
# Find the MPI package
find_package(MPI REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C CXX)


# Set the path to CGNS
set(CGNS_DIR /opt/cgns)
# We won't set HDF5_DIR here, allowing CMake to automatically find HDF5

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

# Find OpenCV package
find_package(OpenCV REQUIRED)

set(OPENCV_LIBS
    opencv_cudaimgproc
    opencv_cudacodec
    opencv_cudafilters
    opencv_cudalegacy
    opencv_stitching
    opencv_aruco
    opencv_bgsegm
    opencv_bioinspired
    opencv_ccalib
    opencv_dnn_objdetect
    opencv_dnn_superres
    opencv_dpm
    opencv_highgui
    opencv_face
    opencv_freetype
    opencv_fuzzy
    opencv_hdf
    opencv_hfs
    opencv_img_hash
    opencv_line_descriptor
    opencv_quality
    opencv_reg
    opencv_rgbd
    opencv_saliency
    opencv_shape
    opencv_stereo
    opencv_structured_light
    opencv_phase_unwrapping
    opencv_superres
    opencv_optflow
    opencv_surface_matching
    opencv_tracking
    opencv_datasets
    opencv_text
    opencv_dnn
    opencv_plot
    opencv_ml
    opencv_videostab
    opencv_videoio
    opencv_viz
    opencv_ximgproc
    opencv_video
    opencv_xobjdetect
    opencv_objdetect
    opencv_calib3d
    opencv_imgcodecs
    opencv_features2d
    opencv_flann
    opencv_xphoto
    opencv_photo
    opencv_imgproc
    opencv_core
)

link_directories(${CGNS_DIR}/lib ${Python3_LIBRARY_DIRS} /usr/local/lib)

# Specify the include directories for your project
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/solve
    ${PROJECT_SOURCE_DIR}/include/setup
    ${CGNS_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
    ${CUDA_TOOLKIT_INCLUDE}
    ${Python3_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

# Set CUDA flags (adjust according to your GPU architecture)
set(CUDA_ARCH_BIN "61")  # Example: for NVIDIA T4 GPU, change as needed
set(CUDA_ARCH_PTX "61")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -gencode arch=compute_${CUDA_ARCH_BIN},code=sm_${CUDA_ARCH_BIN}")

# Add source files for your project
set(SOURCES
    src/app/main.cpp
)

message(SOURCES)

# Enable language CUDA (allow CMake to handle CUDA compilation)
enable_language(CUDA)

# Define the PROJECT_PATH macro for use in the C++ code
if(DEFINED PROJECT_PATH)
    add_definitions(-DPROJECT_PATH="${PROJECT_PATH}")
endif()

# Create the executable with the CUDA compiler (nvcc) for CUDA code in C++ file
add_executable(${PROJECT_NAME} ${SOURCES})

# Specify that we are linking with CUDA libraries
target_link_libraries(${PROJECT_NAME}
    ${CUDA_LIBRARIES}  # Link to CUDA libraries
    cgns
    MPI::MPI_CXX
    HDF5::HDF5
    Python3::Python
    pybind11::module
    ${OPENCV_LIBS}
    ${OpenCV_LIBS}
)

# Set the output directory for the compiled binary
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

# Enable Debug and Release flags based on build type
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message("Building in Debug mode")
    # Add additional debug flags
    target_compile_options(${PROJECT_NAME} PRIVATE -g -Wall -DDEBUG)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    message("Building in Release mode")
    # Add additional release flags
    target_compile_options(${PROJECT_NAME} PRIVATE -O3)
endif()

# Automatically set CMake build type if not provided (default to Debug)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build." FORCE)
    message("Setting default build type to Debug")
endif()
