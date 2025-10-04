cmake_minimum_required(VERSION 3.18)  # Use >= 3.18 for proper CUDA support

project(medsolver LANGUAGES C CXX CUDA)



# --- C++ Standard ---
#set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")

# --- Output paths ---
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# --- Compiler Warnings ---
add_compile_options(-Wall -Wextra -Wpedantic)

# --- Include headers ---
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/extern/eigen)

# --- Add CUDA static library ---
add_subdirectory(cuda)  # Expects cuda/CMakeLists.txt that builds cuda_matrix_ops

# --- Gather host (C++) source files ---
file(GLOB_RECURSE SRC_FILES ${CMAKE_SOURCE_DIR}/src/*.cpp)

# --- Build the main executable ---
add_executable(${PROJECT_NAME} ${SRC_FILES})

# --- Link CUDA static lib ---
target_link_libraries(${PROJECT_NAME} PRIVATE cuda_matrix_ops)


set_target_properties(medsolver PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# --- VTK Setup ---
find_package(VTK REQUIRED COMPONENTS
    CommonCore
    IOImage
    IOLegacy
    FiltersCore
    FiltersGeneral
    FiltersGeometry
    CommonDataModel
    CommonExecutionModel
    InteractionStyle
    RenderingCore
    RenderingOpenGL2
    IOXML
    ImagingCore
    FiltersModeling
    FiltersSources
    ImagingGeneral
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    VTK::CommonCore
    VTK::IOImage
    VTK::IOLegacy
    VTK::IOXML
    VTK::ImagingCore
    VTK::FiltersCore
    VTK::FiltersGeneral
    VTK::FiltersGeometry
    VTK::CommonDataModel
    VTK::CommonExecutionModel
    VTK::InteractionStyle
    VTK::RenderingCore
    VTK::RenderingOpenGL2
    VTK::FiltersModeling
    VTK::FiltersSources
    VTK::ImagingGeneral
)

# --- vtkbone Setup ---
find_package(vtkbone REQUIRED)

if(NOT vtkbone_FOUND)
    message(FATAL_ERROR "vtkbone not found. Set -Dvtkbone_DIR=/path/to/vtkbone/cmake")
endif()

target_include_directories(${PROJECT_NAME} PRIVATE /opt/VTKbone/include)
target_link_libraries(${PROJECT_NAME} PRIVATE /opt/VTKbone/lib/libvtkbone.so)


find_package(CUDAToolkit REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE
    cuda_matrix_ops
    CUDA::cudart
    pthread
    rt
    dl
)

# --- If GCC 14 is custom installed ---
link_directories(/opt/gcc-14/lib64)
