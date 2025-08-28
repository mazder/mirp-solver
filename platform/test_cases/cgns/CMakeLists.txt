# Set the minimum CMake version required
cmake_minimum_required(VERSION 3.10)

# Set the project name and version
project(CGNSExample VERSION 1.0)

# Set the path to CGNS
set(CGNS_DIR /opt/cgns)

# Add the CGNS include directory to the project
include_directories(${CGNS_DIR}/include)  # Correct path to include directory

# Add the CGNS library directory to the project
link_directories(${CGNS_DIR}/lib)

# Add executable (your C source file)
add_executable(main main.cpp)

# Link CGNS library with the executable
target_link_libraries(main cgns)
