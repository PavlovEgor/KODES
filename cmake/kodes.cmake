# The KODES library, as a source list and an include list.
#
# Every method is compiled in, whichever one a given run ends up naming: the
# choice is made when the program starts, so the code for all of them has to be
# in the binary. That is what the table in method_table.cu refers to.
#
# Include with
#   include(${CMAKE_CURRENT_SOURCE_DIR}/<path to>/cmake/kodes.cmake)
# and then use ${KODES_SOURCES} / ${KODES_INCLUDE_DIRS}.
#
# The pyJac mechanism is a separate list, since not every target needs one:
#   kodes_pyjac_mechanism(grimech MECH_SOURCES MECH_INCLUDE_DIRS)
#
# And the whole of it, library and mechanism, as one prebuilt shared library
# that a program without a CUDA compiler opens with dlopen():
#   kodes_library(grimech)

set(KODES_ROOT_DIR ${CMAKE_CURRENT_LIST_DIR}/..)
set(KODES_SRC_DIR ${KODES_ROOT_DIR}/src)

file(STRINGS ${KODES_ROOT_DIR}/VERSION KODES_VERSION LIMIT_COUNT 1)
string(STRIP "${KODES_VERSION}" KODES_VERSION)

set(KODES_SOURCES
    ${KODES_SRC_DIR}/basic_linalg.cu

    ${KODES_SRC_DIR}/Resources/StepState.cu
    ${KODES_SRC_DIR}/Resources/HostResources.cu
    ${KODES_SRC_DIR}/Resources/DeviceResources.cu
    ${KODES_SRC_DIR}/Resources/AdaptiveDeviceResources.cu
    ${KODES_SRC_DIR}/Resources/Operator.cu

    ${KODES_SRC_DIR}/Integrator/Integrator.cu
    ${KODES_SRC_DIR}/Integrator/IntegrationMethods/IntegrationMethod.cu
    ${KODES_SRC_DIR}/Integrator/IntegrationMethods/method_table.cu
    ${KODES_SRC_DIR}/Integrator/IntegrationMethods/Seulex/Seulex.cu
    ${KODES_SRC_DIR}/Integrator/IntegrationMethods/Seulex/SeulexDeviceResources.cu
    ${KODES_SRC_DIR}/Integrator/IntegrationMethods/Seulex/seulex_constants.cu
    ${KODES_SRC_DIR}/Integrator/IntegrationMethods/Euler/Euler.cu
    ${KODES_SRC_DIR}/Integrator/IntegrationMethods/Euler/EulerDeviceResources.cu

    ${KODES_SRC_DIR}/ODESystem/PyJacSystem/PyJacSystem.cu
    ${KODES_SRC_DIR}/ODESystem/system_table.cu
)

# One entry per family root, not per class: a concrete class is included
# through its directory, "Seulex/Seulex.cuh", so adding one
# never touches this list
set(KODES_INCLUDE_DIRS
    ${KODES_SRC_DIR}
    ${KODES_SRC_DIR}/Factory
    ${KODES_SRC_DIR}/ODESystem
    ${KODES_SRC_DIR}/Resources
    ${KODES_SRC_DIR}/Integrator
    ${KODES_SRC_DIR}/Integrator/IntegrationMethods
    ${KODES_SRC_DIR}/Settings
)

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
#
# A mechanism kept somewhere else - a consumer's own - is named by the
# directory holding pyJac's out/, the layout data/mechanisms/ uses too:
#   kodes_pyjac_mechanism(C1_C3 MECH_SOURCES MECH_INCLUDE_DIRS
#                         DIRECTORY /path/to/C1_C3)
function(kodes_pyjac_mechanism name out_sources out_include_dirs)

    cmake_parse_arguments(arg "" "DIRECTORY" "" ${ARGN})

    if(arg_DIRECTORY)
        set(dir ${arg_DIRECTORY}/out)
    else()
        set(dir ${KODES_MECHANISM_DIR}/${name}/out)
    endif()

    if(NOT EXISTS ${dir}/mechanism.cuh)
        message(FATAL_ERROR
            "kodes_pyjac_mechanism: no pyJac mechanism \"${name}\" in ${dir}")
    endif()

    # PyJacSystem.cu and system_table.cu are in the core list, since neither
    # names a mechanism - they take whichever one is on the include path
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

# The specie names of a pyJac mechanism, in its own order, written into
# <out_file> as the macro KODES_MECHANISM_SPECIES - a comma separated list of
# string literals.
#
# pyJac writes them nowhere but a comment of mechanism.cuh, the "Species
# Indexes" block, one "<index>  <name>" per line, so this reads that block the
# way the OpenFOAM chemistry model used to read it at run time. The list is
# checked against NSP where it is used, in Library/library_api.cu.
function(_kodes_write_species_header mechanism_dir out_file)

    set(header ${mechanism_dir}/mechanism.cuh)

    file(READ ${header} text)

    string(FIND "${text}" "Species Indexes" begin)

    if(begin EQUAL -1)
        message(FATAL_ERROR "${header} has no \"Species Indexes\" block")
    endif()

    string(SUBSTRING "${text}" ${begin} -1 text)
    string(FIND "${text}" "*/" end)
    string(SUBSTRING "${text}" 0 ${end} text)

    string(REPLACE "\n" ";" lines "${text}")

    set(entries "")

    foreach(line IN LISTS lines)
        if(line MATCHES "^[ \t]*[0-9]+[ \t]+([^ \t\r]+)[ \t\r]*$")
            string(APPEND entries "    \"${CMAKE_MATCH_1}\", \\\n")
        endif()
    endforeach()

    # only rewritten when the list changes, so a reconfigure rebuilds nothing
    file(CONFIGURE OUTPUT ${out_file} CONTENT
"// Generated by cmake/kodes.cmake from ${header} - do not edit.
#define KODES_MECHANISM_SPECIES \\
${entries}
" @ONLY)

    # a regenerated mechanism has to reach the list
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${header})

endfunction()

# libkodes_<name>.so: the whole library and one pyJac mechanism behind a single
# device link, for a program that has no CUDA compiler to open with dlopen().
#
#   kodes_library(grimech)                            data/mechanisms/grimech
#   kodes_library(C1_C3 DIRECTORY /path/to/C1_C3)     a mechanism of your own
#
# Why it has to be one library per mechanism: the solve kernel calls the
# method and the mechanism through device-only virtuals, and
# those are resolved at device link time. CUDA does not link device code across
# shared libraries, so the device link has to see all of it at once - mechanism
# included - and it is done here, when the .so is built
# (CUDA_RESOLVE_DEVICE_SYMBOLS). What the .so offers in return is host code
# only: Library/library_api.h, which compiles with any C++ compiler.
#
# The CUDA runtime is linked in statically, so the .so needs the driver and
# nothing else of CUDA on the machine it runs on. Every symbol but the two
# entry points of library_api.h is hidden, its own and the runtime's alike, so
# two of these opened into one process cannot see each other.
function(kodes_library name)

    cmake_parse_arguments(arg "" "DIRECTORY" "" ${ARGN})

    if(arg_DIRECTORY)
        set(mechanism_dir ${arg_DIRECTORY}/out)
        kodes_pyjac_mechanism(${name} mech_sources mech_include_dirs
                              DIRECTORY ${arg_DIRECTORY})
    else()
        set(mechanism_dir ${KODES_MECHANISM_DIR}/${name}/out)
        kodes_pyjac_mechanism(${name} mech_sources mech_include_dirs)
    endif()

    set(target kodes_${name})
    set(generated_dir ${CMAKE_CURRENT_BINARY_DIR}/${target})

    _kodes_write_species_header(${mechanism_dir} ${generated_dir}/mechanism_species.h)

    add_library(${target} SHARED
        ${KODES_SOURCES}
        ${KODES_SRC_DIR}/Library/library_api.cu
        ${mech_sources}
    )

    target_include_directories(${target} PRIVATE
        ${KODES_INCLUDE_DIRS}
        ${KODES_SRC_DIR}/Library
        ${mech_include_dirs}
        ${generated_dir}
    )

    target_compile_definitions(${target} PRIVATE
        KODES_MECHANISM_NAME="${name}"
        KODES_VERSION="${KODES_VERSION}"
    )

    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fvisibility=hidden>
    )

    set_target_properties(${target} PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_RUNTIME_LIBRARY Static
    )

    # --no-undefined: a pyJac source missing from the list fails the build
    # here, not the dlopen() of a case hours later
    target_link_options(${target} PRIVATE
        "LINKER:--no-undefined"
        "LINKER:--exclude-libs,ALL"
    )

endfunction()
