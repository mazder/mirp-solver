# Set minimum required version of CMake
cmake_minimum_required(VERSION 3.10)

# Set the project name and specify the required C++ standard
project(JsonToCGNSExample CXX)
set(CMAKE_CXX_STANDARD 11)

# Specify the directory where CGNS is installed
set(CGNS_DIR /opt/cgns)

# Include the CGNS headers
include_directories(${CGNS_DIR}/include)

# Set the location for the CGNS library
link_directories(${CGNS_DIR}/lib)

# Add the executable for your project
add_executable(json_to_cgns main.cpp)

# Link the CGNS library with the executable
target_link_libraries(json_to_cgns cgns)
