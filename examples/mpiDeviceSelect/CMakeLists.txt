cmake_minimum_required(VERSION 3.15)
project(KodesMpiDeviceSelect LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../..)

include(${ROOT_DIR}/cmake/kodes.cmake)

# для RTX 20xx, 30xx, 40xx, 50xx
set(CMAKE_CUDA_ARCHITECTURES "120")

find_package(MPI REQUIRED)

# only the device binding helper, none of the solve machinery
add_executable(mpiDeviceSelect mpiDeviceSelect.cu ${KODES_MPI_SOURCES})

target_include_directories(mpiDeviceSelect PRIVATE ${KODES_SRC_DIR})

target_link_libraries(mpiDeviceSelect PRIVATE MPI::MPI_CXX)
