cmake_minimum_required(VERSION 3.15)
project(KodesReactors LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../..)

include(${ROOT_DIR}/cmake/kodes.cmake)

# для RTX 20xx, 30xx, 40xx, 50xx
set(CMAKE_CUDA_ARCHITECTURES "120")

# One program, one target per mechanism. The mechanism is the only thing that
# has to be fixed at build time - pyJac generates C for one of them and NSP is
# a macro of its mechanism.cuh. The method, the balancer and the tolerances are
# names and numbers in the .json beside this file.
function(kodes_reactor_case mechanism)

    set(target reactors_${mechanism})

    kodes_pyjac_mechanism(${mechanism} MECH_SOURCES MECH_INCLUDE_DIRS)

    add_executable(${target}
        reactors.cu
        ${KODES_SOURCES}
        ${KODES_SETTINGS_SOURCES}
        ${MECH_SOURCES}
    )

    target_include_directories(${target} PRIVATE
        ${KODES_INCLUDE_DIRS}
        ${KODES_SETTINGS_INCLUDE_DIRS}
        ${MECH_INCLUDE_DIRS}
    )

    target_compile_definitions(${target} PRIVATE
        KODES_MECHANISM_NAME="${mechanism}"
        KODES_DEFAULT_SETTINGS="${mechanism}.json"
    )

    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math>
    )

    # a device-only virtual is called from a kernel in another translation
    # unit, so the device link step is not optional
    set_target_properties(${target} PROPERTIES
        CUDA_ARCHITECTURES "120"
        CUDA_SEPARABLE_COMPILATION ON
    )

    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-G -lineinfo>
        )
    endif()

    # each case reads its own settings by name from the working directory
    configure_file(
        ${mechanism}.json
        ${CMAKE_CURRENT_BINARY_DIR}/${mechanism}.json
        COPYONLY
    )

endfunction()

# add a line to add a case; the mechanism must be a directory of
# data/mechanisms
kodes_reactor_case(grimech)
kodes_reactor_case(h2o2)
