cmake_minimum_required(VERSION 3.15)
project(MFEMExample1 LANGUAGES CXX)

# Use C++17 (required by MFEM)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set 64-bit METIS index/real types
add_compile_definitions(IDXTYPEWIDTH=64)
add_compile_definitions(REALTYPEWIDTH=64)

# Specify MFEM installation directory
set(MFEM_DIR "/opt/mfem")  # Update if MFEM is elsewhere

# Include MFEM headers
include_directories(${MFEM_DIR})
link_directories(${MFEM_DIR})

# Locate METIS
find_path(METIS_INCLUDE_DIR metis.h)
find_library(METIS_LIBRARY NAMES metis)

if(NOT METIS_INCLUDE_DIR OR NOT METIS_LIBRARY)
    message(FATAL_ERROR "METIS not found. Make sure libmetis-dev is installed.")
endif()
include_directories(${METIS_INCLUDE_DIR})

# Add executable
add_executable(ex1 ex1.cpp)

# Link libraries: MFEM first, METIS after (important!)
target_link_libraries(ex1 PRIVATE mfem ${METIS_LIBRARY})

# Link pthread if required by MFEM
find_package(Threads REQUIRED)
target_link_libraries(ex1 PRIVATE Threads::Threads)
