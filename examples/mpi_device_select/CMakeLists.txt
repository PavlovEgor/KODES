cmake_minimum_required(VERSION 3.15)
project(MpiDeviceSelectExample LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURES "120")  # для RTX 20xx, 30xx, 40xx, 50xx

find_package(MPI REQUIRED)

add_executable(mpi_device_select
    main.cu
    ../../src/kodes_mpi.cu
)

target_include_directories(mpi_device_select PRIVATE
    ../../include
)

target_link_libraries(mpi_device_select PRIVATE MPI::MPI_CXX)
