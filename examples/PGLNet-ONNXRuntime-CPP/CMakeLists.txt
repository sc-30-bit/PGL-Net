cmake_minimum_required(VERSION 3.5)
project(PGLNetOnnxRuntimeCPP VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

option(USE_CUDA "Enable CUDA EP" ON)
if(NOT APPLE AND USE_CUDA)
    add_definitions(-DUSE_CUDA)
endif()

if(NOT DEFINED ONNXRUNTIME_ROOT OR ONNXRUNTIME_ROOT STREQUAL "")
    message(FATAL_ERROR "Please set -DONNXRUNTIME_ROOT=/path/to/onnxruntime")
endif()

include_directories(${ONNXRUNTIME_ROOT}/include)

set(PROJECT_SOURCES
    main.cpp
    inference.h
    inference.cpp
)

add_executable(pglnet_onnxruntime ${PROJECT_SOURCES})

if(WIN32)
    target_link_libraries(pglnet_onnxruntime PRIVATE ${OpenCV_LIBS} ${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib)
    add_custom_command(TARGET pglnet_onnxruntime POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
                "$<TARGET_FILE_DIR:pglnet_onnxruntime>/onnxruntime.dll")
    if(USE_CUDA)
        add_custom_command(TARGET pglnet_onnxruntime POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_shared.dll"
                    "$<TARGET_FILE_DIR:pglnet_onnxruntime>/onnxruntime_providers_shared.dll"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_cuda.dll"
                    "$<TARGET_FILE_DIR:pglnet_onnxruntime>/onnxruntime_providers_cuda.dll")
    endif()
elseif(APPLE)
    target_link_libraries(pglnet_onnxruntime PRIVATE ${OpenCV_LIBS} ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.dylib)
else()
    target_link_libraries(pglnet_onnxruntime PRIVATE ${OpenCV_LIBS} ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so)
endif()
