cmake_minimum_required(VERSION 3.10)

project(CONV2D)

set(CMAKE_CXX_STANDARD 11)

add_library(conv2d
    src/conv2d.cpp
    src/conv2d.h
    src/Mat.cpp
    src/Mat.h
)

target_include_directories(conv2d PUBLIC src/)

add_executable(test_main test_main.cpp)

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Turn off the fking test!")
add_subdirectory(benchmark)

target_include_directories(test_main PUBLIC src/)
target_link_libraries(test_main conv2d benchmark::benchmark)
