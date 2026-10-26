cmake_minimum_required(VERSION 3.15)

# No CUDA in this project: it opens a libkodes_<mechanism>.so built by the
# CMakeLists.txt at the top of the repository, and includes nothing of KODES
# but library_api.h.
project(KodesLibrary LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../..)

add_executable(library library.cpp)

target_include_directories(library PRIVATE ${ROOT_DIR}/src/Library)

target_link_libraries(library PRIVATE ${CMAKE_DL_LIBS})
