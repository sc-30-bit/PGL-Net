cmake_minimum_required(VERSION 3.5)
project(PGLNetOpenCVDNNCPP LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(pglnet_opencv_dnn
    main.cpp
    inference.h
    inference.cpp
)
target_link_libraries(pglnet_opencv_dnn PRIVATE ${OpenCV_LIBS})

