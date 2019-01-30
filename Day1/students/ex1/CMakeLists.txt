cmake_minimum_required(VERSION 2.8.3)
set(EXECUTABLE_NAME Labwork2)
set(PROJECT_NAME Labwork2)

# set CUDA
find_package(CUDA REQUIRED)  
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -gencode arch=compute_20,code=sm_20) 

cuda_compile(CUDA_OBJECTS main.cu Student.cu D_Matrix.cu H_Matrix.cu utils/chronoGPU.cu) 

project(${EXECUTABLE_NAME} LANGUAGES CXX )


# Tell CMake to run moc when necessary
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
	
include_directories(include "D:/CUDA Samples/v10.0/common/inc")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EXTRA_CXX_FLAGS}")  
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --gpu-architecture=sm_52")

set(HEADERS D_Matrix.cuh H_Matrix.cuh utils/chronoCPU.hpp utils/chronoGPU.hpp)
set(SOURCES  utils/chronoCPU.cpp)

#Remember dependencies!  
add_executable(${EXECUTABLE_NAME} ${SOURCES} ${HEADERS} ${CUDA_OBJECTS})
target_link_libraries(${EXECUTABLE_NAME})

# Enable C++14 standard compliance
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_STANDARD 14)
