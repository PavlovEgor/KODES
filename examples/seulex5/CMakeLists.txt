cmake_minimum_required(VERSION 3.15)
project(KodesSeulex5 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../..)

include(${ROOT_DIR}/cmake/kodes.cmake)

kodes_pyjac_mechanism(grimech MECH_SOURCES MECH_INCLUDE_DIRS)

# для RTX 20xx, 30xx, 40xx, 50xx
set(CMAKE_CUDA_ARCHITECTURES "120")

add_executable(seulex5
    seulex5.cu
    ${KODES_SOURCES}
    ${KODES_SETTINGS_SOURCES}
    ${MECH_SOURCES}
)

target_include_directories(seulex5 PRIVATE
    ${KODES_INCLUDE_DIRS}
    ${KODES_SETTINGS_INCLUDE_DIRS}
    ${MECH_INCLUDE_DIRS}
)

# the run reads it by name from the working directory
configure_file(seulex5.json ${CMAKE_CURRENT_BINARY_DIR}/seulex5.json COPYONLY)

target_compile_options(seulex5 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math>
)

# a device-only virtual is called from a kernel in another translation unit, so
# the device link step is not optional
set_target_properties(seulex5 PROPERTIES
    CUDA_ARCHITECTURES "120"
    CUDA_SEPARABLE_COMPILATION ON
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(seulex5 PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-G -lineinfo>
    )
endif()
