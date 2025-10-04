cmake_minimum_required(VERSION 3.10)
project(main)

find_package(VTK REQUIRED)

add_executable(main main.cpp)
target_link_libraries(main PRIVATE ${VTK_LIBRARIES})
