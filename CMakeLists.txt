cmake_minimum_required(VERSION 3.18)

# The prebuilt libraries: libkodes_<mechanism>.so, one per pyJac mechanism, for
# a program that opens them with dlopen() and has no CUDA compiler of its own.
# See "As a prebuilt library" in the ReadMe.
#
#   cmake -B build
#   cmake --build build -j
#   cmake --install build
#
# The examples are separate projects, each building KODES from source.

# sm_120, the RTX 50xx, as in the examples; -DCMAKE_CUDA_ARCHITECTURES="80;90"
# for other cards
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 120)
endif()

project(KODES LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/kodes.cmake)

# Every mechanism of data/mechanisms/ unless told otherwise
file(GLOB entries LIST_DIRECTORIES true ${KODES_MECHANISM_DIR}/*)

set(default_mechanisms "")

foreach(entry IN LISTS entries)
    if(IS_DIRECTORY ${entry})
        get_filename_component(name ${entry} NAME)
        list(APPEND default_mechanisms ${name})
    endif()
endforeach()

# An entry is either the name of a directory of data/mechanisms/ or the path of
# a directory holding pyJac's out/, a relative one taken from this directory.
# A path builds libkodes_<its last component>.so.
set(KODES_MECHANISMS "${default_mechanisms}" CACHE STRING
    "The mechanisms to build a libkodes_<name>.so for")

# Where `cmake --install` puts them. With an OpenFOAM environment active that is
# its user library directory: it is on LD_LIBRARY_PATH, so dlopen() finds the
# libraries there by name, which is how kodesChemistryModel looks for them.
if(DEFINED ENV{FOAM_USER_LIBBIN})
    set(default_library_dir $ENV{FOAM_USER_LIBBIN})
else()
    set(default_library_dir lib)
endif()

set(KODES_LIBRARY_DIR "${default_library_dir}" CACHE PATH
    "Where to install the libkodes_<mechanism>.so libraries")

foreach(mechanism IN LISTS KODES_MECHANISMS)

    if(mechanism MATCHES "/")
        get_filename_component(dir ${mechanism} ABSOLUTE
                               BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
        get_filename_component(name ${dir} NAME)
        kodes_library(${name} DIRECTORY ${dir})
    else()
        set(name ${mechanism})
        kodes_library(${name})
    endif()

    install(TARGETS kodes_${name} LIBRARY DESTINATION ${KODES_LIBRARY_DIR})

endforeach()
