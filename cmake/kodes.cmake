# The KODES library, as a source list and an include list.
#
# Every method and every balancer is compiled in, whichever one a given run
# ends up naming: the choice is made when the program starts, so the code for
# all of them has to be in the binary. That is what the tables in methodTable.cu
# and balancerTable.cu refer to.
#
# Include with
#   include(${CMAKE_CURRENT_SOURCE_DIR}/<path to>/cmake/kodes.cmake)
# and then use ${KODES_SOURCES} / ${KODES_INCLUDE_DIRS}.
#
# The pyJac mechanism is a separate list, since not every target needs one:
#   kodes_pyjac_mechanism(grimech OUT_SOURCES ... OUT_INCLUDE_DIRS ...)

set(KODES_ROOT_DIR ${CMAKE_CURRENT_LIST_DIR}/..)
set(KODES_SRC_DIR ${KODES_ROOT_DIR}/src)

set(KODES_SOURCES
    ${KODES_SRC_DIR}/basic_linalg.cu

    ${KODES_SRC_DIR}/StepState/StepState.cu

    ${KODES_SRC_DIR}/Resources/DeviceResources.cu
    ${KODES_SRC_DIR}/Resources/AdaptiveDeviceResources.cu
    ${KODES_SRC_DIR}/Resources/HostResources.cu
    ${KODES_SRC_DIR}/Resources/Operator.cu
    ${KODES_SRC_DIR}/Resources/IntegratorDeviceResources/Seulex/SeulexDeviceResources.cu
    ${KODES_SRC_DIR}/Resources/IntegratorDeviceResources/Seulex/SeulexConstants.cu
    ${KODES_SRC_DIR}/Resources/IntegratorDeviceResources/Euler/EulerDeviceResources.cu

    ${KODES_SRC_DIR}/Integrator/Integrator.cu
    ${KODES_SRC_DIR}/Integrator/IntegrationMethods/IntegrationMethod.cu
    ${KODES_SRC_DIR}/Integrator/IntegrationMethods/methodTable.cu
    ${KODES_SRC_DIR}/Integrator/IntegrationMethods/Seulex/Seulex.cu
    ${KODES_SRC_DIR}/Integrator/IntegrationMethods/Euler/Euler.cu

    ${KODES_SRC_DIR}/Balancer/Balancer.cu
    ${KODES_SRC_DIR}/Balancer/TemperatureBalancer.cu
    ${KODES_SRC_DIR}/Balancer/RHSNormBalancer.cu
    ${KODES_SRC_DIR}/Balancer/StiffnessBalancer.cu
    ${KODES_SRC_DIR}/Balancer/balancerTable.cu
)

set(KODES_INCLUDE_DIRS
    ${KODES_SRC_DIR}
    ${KODES_SRC_DIR}/Factory
    ${KODES_SRC_DIR}/StepState
    ${KODES_SRC_DIR}/ODESystem
    ${KODES_SRC_DIR}/Resources
    ${KODES_SRC_DIR}/Resources/IntegratorDeviceResources/Seulex
    ${KODES_SRC_DIR}/Resources/IntegratorDeviceResources/Euler
    ${KODES_SRC_DIR}/Integrator
    ${KODES_SRC_DIR}/Integrator/IntegratorControls
    ${KODES_SRC_DIR}/Integrator/IntegrationMethods
    ${KODES_SRC_DIR}/Integrator/IntegrationMethods/Seulex
    ${KODES_SRC_DIR}/Integrator/IntegrationMethods/Euler
    ${KODES_SRC_DIR}/Balancer
)

# The generated sources of one pyJac mechanism, plus pyJacSystem itself.
function(kodes_pyjac_mechanism name out_sources out_include_dirs)

    set(dir ${KODES_SRC_DIR}/ODESystem/${name}/out)

    set(sources
        ${KODES_SRC_DIR}/ODESystem/pyJacSystem.cu
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
