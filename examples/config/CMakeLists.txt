cmake_minimum_required(VERSION 3.15)
project(KodesConfig LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../..)

include(${ROOT_DIR}/cmake/kodes.cmake)

# для RTX 20xx, 30xx, 40xx, 50xx
set(CMAKE_CUDA_ARCHITECTURES "120")

# kodes::Config on its own: no device object is built here, so none of
# KODES_SOURCES is needed
add_executable(config config.cu ${KODES_SRC_DIR}/Settings/Config.cu)

target_include_directories(config PRIVATE
    ${KODES_SRC_DIR}
    ${KODES_SRC_DIR}/Settings
    ${KODES_SETTINGS_INCLUDE_DIRS}
)

configure_file(config.json ${CMAKE_CURRENT_BINARY_DIR}/config.json COPYONLY)
