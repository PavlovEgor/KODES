cmake_minimum_required(VERSION 3.15)
project(KodesEuler1 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../..)

include(${ROOT_DIR}/cmake/kodes.cmake)

kodes_pyjac_mechanism(grimech MECH_SOURCES MECH_INCLUDE_DIRS)

# для RTX 20xx, 30xx, 40xx, 50xx
set(CMAKE_CUDA_ARCHITECTURES "120")

add_executable(euler1 euler1.cu ${KODES_SOURCES} ${MECH_SOURCES})

target_include_directories(euler1 PRIVATE
    ${KODES_INCLUDE_DIRS}
    ${MECH_INCLUDE_DIRS}
)

target_compile_options(euler1 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math>
)

set_target_properties(euler1 PROPERTIES
    CUDA_ARCHITECTURES "120"
    CUDA_SEPARABLE_COMPILATION ON
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(euler1 PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-G -lineinfo>
    )
endif()
