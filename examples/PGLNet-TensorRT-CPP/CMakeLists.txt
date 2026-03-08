cmake_minimum_required(VERSION 3.12)
project(PGLNetTensorRTCPP LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

if(NOT DEFINED TensorRT_ROOT)
    message(FATAL_ERROR "Please set -DTensorRT_ROOT=/path/to/TensorRT")
endif()

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${TensorRT_ROOT}/include
)

add_executable(pglnet_tensorrt
    main.cpp
    inference.h
    inference.cpp
)

if(WIN32)
    target_link_libraries(pglnet_tensorrt PRIVATE
        ${OpenCV_LIBS}
        ${CUDA_LIBRARIES}
        ${TensorRT_ROOT}/lib/nvinfer.lib
        ${TensorRT_ROOT}/lib/cudart.lib
    )
else()
    target_link_libraries(pglnet_tensorrt PRIVATE
        ${OpenCV_LIBS}
        ${CUDA_LIBRARIES}
        ${TensorRT_ROOT}/lib/libnvinfer.so
        ${TensorRT_ROOT}/lib/libcudart.so
    )
endif()

