# The KODES library, as a source list and an include list.
#
# Every method and every balancer is compiled in, whichever one a given run
# ends up naming: the choice is made when the program starts, so the code for
# all of them has to be in the binary. That is what the tables in method_table.cu
# and balancer_table.cu refer to.
#
# Include with
#   include(${CMAKE_CURRENT_SOURCE_DIR}/<path to>/cmake/kodes.cmake)
# and then use ${KODES_SOURCES} / ${KODES_INCLUDE_DIRS}.
#
# The lists themselves are NOT here. They live in wmake/kodes.files and
# wmake/kodes.options, in the form OpenFOAM's wmake wants, and this file reads
# them - so a source or an include directory added to KODES reaches CMake and
# wmake from one place. See "Building against KODES" in the ReadMe.
#
# The pyJac mechanism is a separate list, since not every target needs one:
#   kodes_pyjac_mechanism(grimech MECH_SOURCES MECH_INCLUDE_DIRS)

set(KODES_ROOT_DIR ${CMAKE_CURRENT_LIST_DIR}/..)
set(KODES_SRC_DIR ${KODES_ROOT_DIR}/src)
set(KODES_WMAKE_DIR ${KODES_ROOT_DIR}/wmake)

# Read one of the wmake fragments: drop the C comment block and the blank
# lines, turn $(KODES_SRC) into this build's path, and hand back a CMake list.
function(_kodes_read_wmake_list file prefix_to_strip out_list)

    file(READ ${file} text)

    # the /* ... */ header
    string(REGEX REPLACE "/\\*.*\\*/" "" text "${text}")

    string(REPLACE "$(KODES_SRC)" "${KODES_SRC_DIR}" text "${text}")
    string(REPLACE "\\" "" text "${text}")
    string(REPLACE "\n" ";" lines "${text}")

    set(result "")

    foreach(line ${lines})
        string(STRIP "${line}" line)

        if(line STREQUAL "")
            continue()
        endif()

        # a make variable definition, not an entry
        if(line MATCHES "=")
            continue()
        endif()

        if(NOT prefix_to_strip STREQUAL "")
            string(REGEX REPLACE "^${prefix_to_strip}" "" line "${line}")
        endif()

        list(APPEND result "${line}")
    endforeach()

    set(${out_list} ${result} PARENT_SCOPE)

endfunction()

_kodes_read_wmake_list(${KODES_WMAKE_DIR}/kodes.files "" KODES_SOURCES)
_kodes_read_wmake_list(${KODES_WMAKE_DIR}/kodes.options "-I" KODES_INCLUDE_DIRS)

# The JSON settings reader, kept apart because it is the one part of the
# library that needs the rapidjson submodule. A caller that gets its settings
# from somewhere else - the OpenFOAM chemistry model reads an OpenFOAM
# dictionary - passes the same names and numbers by hand and never links this.
#
#   git submodule update --init external/rapidjson
set(KODES_SETTINGS_SOURCES
    ${KODES_SRC_DIR}/Settings/Config.cu
    ${KODES_SRC_DIR}/Settings/Settings.cu
)

set(KODES_SETTINGS_INCLUDE_DIRS
    ${KODES_ROOT_DIR}/external/rapidjson/include
)

# Optional MPI device binding, a translation unit of its own so that targets
# not needing MPI never link it
set(KODES_MPI_SOURCES
    ${KODES_SRC_DIR}/mpi_select_device.cu
)


set(KODES_MECHANISM_DIR ${KODES_ROOT_DIR}/data/mechanisms)

# The generated sources of one pyJac mechanism, plus PyJacSystem itself.
#
# The mechanisms are data, not library code: they live under data/mechanisms/,
# each in the directory pyJac wrote it into, together with the input files it
# was generated from. Nothing in src/ mentions any of them by name - PyJacSystem
# calls dydt()/eval_jacob() and NSP comes from whichever mechanism.cuh is on the
# include path - so which one a target uses is decided here and nowhere else.
function(kodes_pyjac_mechanism name out_sources out_include_dirs)

    set(dir ${KODES_MECHANISM_DIR}/${name}/out)

    if(NOT EXISTS ${dir})
        message(FATAL_ERROR
            "kodes_pyjac_mechanism: no mechanism \"${name}\" in "
            "${KODES_MECHANISM_DIR}")
    endif()

    # PyJacSystem.cu itself is in the core list, since it names no mechanism
    set(sources
        ${dir}/chem_utils.cu
        ${dir}/dydt.cu
        ${dir}/jacob.cu
        ${dir}/gpu_memory.cu
        ${dir}/mass_mole.cu
        ${dir}/mechanism.cu
        ${dir}/rxn_rates_pres_mod.cu
        ${dir}/rxn_rates.cu
        ${dir}/sparse_multiplier.cu
        ${dir}/spec_rates.cu
    )

    file(GLOB jacobs ${dir}/jacobs/jacob_*.cu)
    file(GLOB rates ${dir}/rates/rxn_rates_*.cu)

    set(${out_sources} ${sources} ${jacobs} ${rates} PARENT_SCOPE)

    set(${out_include_dirs}
        ${dir}
        ${dir}/jacobs
        ${dir}/rates
        PARENT_SCOPE
    )

endfunction()
