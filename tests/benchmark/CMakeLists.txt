cmake_minimum_required(VERSION 3.15)

# No CUDA in this project: it opens a libkodes_<mechanism>.so built by the
# CMakeLists.txt at the top of the repository, like examples/library. It reads
# case.json with rapidjson:
#   git submodule update --init external/rapidjson
project(KodesBenchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../..)

add_executable(benchmark benchmark.cpp)

target_include_directories(benchmark PRIVATE
    ${ROOT_DIR}/src/Library
    ${ROOT_DIR}/external/rapidjson/include
)

# where the cases are found by name
get_filename_component(tests_dir ${ROOT_DIR}/tests ABSOLUTE)
target_compile_definitions(benchmark PRIVATE KODES_TESTS_DIR="${tests_dir}")

target_link_libraries(benchmark PRIVATE ${CMAKE_DL_LIBS})
