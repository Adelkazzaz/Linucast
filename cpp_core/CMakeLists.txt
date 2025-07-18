cmake_minimum_required(VERSION 3.15)
project(linucastCore)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)

# Optional packages
find_package(Protobuf QUIET)
find_package(ONNXRuntime QUIET)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS} include)

# Create shared library
add_library(linucast SHARED
    src/processor.cpp
    src/kalman.cpp
)

# Link libraries
target_link_libraries(linucast
    ${OpenCV_LIBS}
    Threads::Threads
)

# Optional ONNX Runtime support
if(ONNXRuntime_FOUND)
    target_link_libraries(linucast onnxruntime)
    target_compile_definitions(linucast PRIVATE ONNX_RUNTIME_AVAILABLE)
endif()

# Create standalone executable
add_executable(linucast_main src/main.cpp)
target_link_libraries(linucast_main linucast)

# Python bindings with pybind11 (if available)
find_package(pybind11 QUIET)
if(pybind11_FOUND)
    pybind11_add_module(linucast_cpp src/bridge.cpp)
    target_link_libraries(linucast_cpp PRIVATE linucast)
endif()

# Install targets
install(TARGETS linucast linucast_main
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/ DESTINATION include/linucast)
