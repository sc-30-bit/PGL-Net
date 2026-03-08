cmake_minimum_required(VERSION 3.12)
project(pglnet_mnn_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

# You can pass either:
# 1) -DMNN_ROOT=/path/to/mnn_package
# or
# 2) explicit Windows paths:
#    -DMNN_INCLUDE_DIR=... -DMNN_LIB_DIR=... -DMNN_DLL_PATH=...
if(NOT DEFINED MNN_INCLUDE_DIR)
    if(DEFINED MNN_ROOT)
        set(MNN_INCLUDE_DIR "${MNN_ROOT}/include")
    else()
        set(MNN_INCLUDE_DIR "H:/libraries/mnn_3.4.1_windows_x64_cpu_opencl/mnn_3.4.1_windows_x64_cpu_opencl/include")
    endif()
endif()

if(NOT DEFINED MNN_LIB_DIR)
    if(DEFINED MNN_ROOT)
        set(MNN_LIB_DIR "${MNN_ROOT}/lib/x64/Release/Dynamic/MD")
    else()
        set(MNN_LIB_DIR "H:/libraries/mnn_3.4.1_windows_x64_cpu_opencl/mnn_3.4.1_windows_x64_cpu_opencl/lib/x64/Release/Dynamic/MD")
    endif()
endif()

if(NOT DEFINED MNN_DLL_PATH)
    set(MNN_DLL_PATH "${MNN_LIB_DIR}/MNN.dll")
endif()

if(NOT EXISTS "${MNN_INCLUDE_DIR}")
    message(FATAL_ERROR "MNN include directory not found: ${MNN_INCLUDE_DIR}")
endif()
if(NOT EXISTS "${MNN_LIB_DIR}")
    message(FATAL_ERROR "MNN lib directory not found: ${MNN_LIB_DIR}")
endif()

include_directories(${MNN_INCLUDE_DIR})
link_directories(${MNN_LIB_DIR})

find_library(MNN_LIB NAMES MNN PATHS "${MNN_LIB_DIR}" NO_DEFAULT_PATH)
if(NOT MNN_LIB)
    message(FATAL_ERROR "Cannot find MNN.lib in: ${MNN_LIB_DIR}")
endif()

add_executable(pglnet_mnn main.cpp)
target_include_directories(pglnet_mnn PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(pglnet_mnn PRIVATE ${OpenCV_LIBS} ${MNN_LIB})

# Optional split libs (some MNN packages provide only MNN.lib).
find_library(MNN_EXPRESS_LIB NAMES MNN_Express PATHS "${MNN_LIB_DIR}" NO_DEFAULT_PATH)
if(MNN_EXPRESS_LIB)
    target_link_libraries(pglnet_mnn PRIVATE ${MNN_EXPRESS_LIB})
endif()

find_library(MNN_OPENCV_LIB NAMES MNNOpenCV PATHS "${MNN_LIB_DIR}" NO_DEFAULT_PATH)
if(MNN_OPENCV_LIB)
    target_link_libraries(pglnet_mnn PRIVATE ${MNN_OPENCV_LIB})
endif()

if(WIN32 AND EXISTS "${MNN_DLL_PATH}")
    add_custom_command(TARGET pglnet_mnn POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${MNN_DLL_PATH}"
                "$<TARGET_FILE_DIR:pglnet_mnn>/MNN.dll")
endif()
