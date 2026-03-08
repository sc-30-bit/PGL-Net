cmake_minimum_required(VERSION 3.12)
project(PGLNetOpenVINOCPP LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(OpenVINO REQUIRED)

add_executable(pglnet_openvino
    main.cpp
    inference.h
    inference.cpp
)
target_include_directories(pglnet_openvino PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(pglnet_openvino PRIVATE ${OpenCV_LIBS} openvino::runtime)

