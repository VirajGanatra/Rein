cmake_minimum_required(VERSION 3.28)
project(Rein)

set(CMAKE_CXX_STANDARD 14)


add_subdirectory(extern/pybind11)


include_directories(src/cpp/include)
include_directories(extern/pybind11/include)


pybind11_add_module(replay_buffer
        src/cpp/src/bindings.cpp
        src/cpp/src/replay_buffer.cpp
)


