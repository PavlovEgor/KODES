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
# The pyJac mechanism is a separate list, since not every target needs one:
#   kodes_pyjac_mechanism(grimech MECH_SOURCES MECH_INCLUDE_DIRS)

set(KODES_ROOT_DIR ${CMAKE_CURRENT_LIST_DIR}/..)
set(KODES_SRC_DIR ${KODES_ROOT_DIR}/src)

set(KODES_SOURCES
    ${KODES_SRC_DIR}/basic_linalg.cu

    ${KODES_SRC_DIR}/Resources/StepState.cu
    ${KODES_SRC_DIR}/Resources/HostResources.cu
    ${KODES_SRC_DIR}/Resources/DeviceResources.cu
    ${KODES_SRC_DIR}/Resources/AdaptiveDeviceResources.cu
    ${KODES_SRC_DIR}/Resources/SeulexDeviceResources.cu
    ${KODES_SRC_DIR}/Resources/EulerDeviceResources.cu
    ${KODES_SRC_DIR}/Resources/Operator.cu

    ${KODES_SRC_DIR}/Integrator/Integrator.cu
    ${KODES_SRC_DIR}/Integrator/IntegrationMethods/IntegrationMethod.cu
    ${KODES_SRC_DIR}/Integrator/IntegrationMethods/method_table.cu
    ${KODES_SRC_DIR}/Integrator/IntegrationMethods/Seulex.cu
    ${KODES_SRC_DIR}/Integrator/IntegrationMethods/seulex_constants.cu
    ${KODES_SRC_DIR}/Integrator/IntegrationMethods/Euler.cu

    ${KODES_SRC_DIR}/Balancer/Balancer.cu
    ${KODES_SRC_DIR}/Balancer/TemperatureBalancer.cu
    ${KODES_SRC_DIR}/Balancer/RHSNormBalancer.cu
    ${KODES_SRC_DIR}/Balancer/StiffnessBalancer.cu
    ${KODES_SRC_DIR}/Balancer/balancer_table.cu
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

set(KODES_INCLUDE_DIRS
    ${KODES_SRC_DIR}
    ${KODES_SRC_DIR}/Factory
    ${KODES_SRC_DIR}/ODESystem
    ${KODES_SRC_DIR}/Resources
    ${KODES_SRC_DIR}/Integrator
    ${KODES_SRC_DIR}/Integrator/IntegrationMethods
    ${KODES_SRC_DIR}/Balancer
    ${KODES_SRC_DIR}/Settings
)

# The generated sources of one pyJac mechanism, plus PyJacSystem itself.
function(kodes_pyjac_mechanism name out_sources out_include_dirs)

    set(dir ${KODES_SRC_DIR}/ODESystem/${name}/out)

    set(sources
        ${KODES_SRC_DIR}/ODESystem/PyJacSystem.cu
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
