cmake_minimum_required(VERSION 3.10)

project(CGNS_YML_Project)


# Set the path to CGNS
set(CGNS_DIR /opt/cgns)

# Add the CGNS include directory to the project
include_directories(${CGNS_DIR}/include)  # Correct path to include directory

# Add the CGNS library directory to the project
link_directories(${CGNS_DIR}/lib)

add_executable(main main.cpp)

target_link_libraries(main ${CGNS_LIBRARIES})
include_directories(${CGNS_INCLUDE_DIRS})

find_package(yaml-cpp REQUIRED)
target_link_libraries(main yaml-cpp cgns)
