cmake_minimum_required(VERSION 3.15)
project(GrimechMpiExample LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../)
set(SRC_DIR ${ROOT_DIR}/src)
set(INC_DIR ${ROOT_DIR}/include)

# Убираем предупреждение о CUDA архитектуре
set(CMAKE_CUDA_ARCHITECTURES "120")  # для RTX 20xx, 30xx, 40xx, 50xx

find_package(MPI REQUIRED)

add_executable( seulex3_mpi
    seulex3_mpi.cu
    ${SRC_DIR}/kodes_mpi.cu
    ${SRC_DIR}/basic_linalg.cu
    ${SRC_DIR}/Resources/SeulexDeviceResources.cu
    ${SRC_DIR}/Resources/DeviceResources.cu
    ${SRC_DIR}/Resources/HostResources.cu
    ${SRC_DIR}/ODESystem/pyJacSystem.cu
    ${SRC_DIR}/ODESystem/grimech/out/chem_utils.cu
    ${SRC_DIR}/ODESystem/grimech/out/dydt.cu
    ${SRC_DIR}/ODESystem/grimech/out/jacob.cu
    ${SRC_DIR}/ODESystem/grimech/out/gpu_memory.cu
    ${SRC_DIR}/ODESystem/grimech/out/mass_mole.cu
    ${SRC_DIR}/ODESystem/grimech/out/mechanism.cu
    ${SRC_DIR}/ODESystem/grimech/out/rxn_rates_pres_mod.cu
    ${SRC_DIR}/ODESystem/grimech/out/rxn_rates.cu
    ${SRC_DIR}/ODESystem/grimech/out/sparse_multiplier.cu
    ${SRC_DIR}/ODESystem/grimech/out/spec_rates.cu
    ${SRC_DIR}/ODESystem/grimech/out/jacobs/jacob_0.cu
    ${SRC_DIR}/ODESystem/grimech/out/jacobs/jacob_1.cu
    ${SRC_DIR}/ODESystem/grimech/out/jacobs/jacob_2.cu
    ${SRC_DIR}/ODESystem/grimech/out/jacobs/jacob_3.cu
    ${SRC_DIR}/ODESystem/grimech/out/jacobs/jacob_4.cu
    ${SRC_DIR}/ODESystem/grimech/out/jacobs/jacob_5.cu
    ${SRC_DIR}/ODESystem/grimech/out/jacobs/jacob_6.cu
    ${SRC_DIR}/ODESystem/grimech/out/jacobs/jacob_7.cu
    ${SRC_DIR}/ODESystem/grimech/out/jacobs/jacob_8.cu
    ${SRC_DIR}/ODESystem/grimech/out/jacobs/jacob_9.cu
    ${SRC_DIR}/ODESystem/grimech/out/jacobs/jacob_10.cu
    ${SRC_DIR}/ODESystem/grimech/out/rates/rxn_rates_0.cu
    ${SRC_DIR}/ODESystem/grimech/out/rates/rxn_rates_1.cu
    )

target_include_directories(seulex3_mpi PRIVATE
    ${INC_DIR}
    ${INC_DIR}/Resources
    ${INC_DIR}/ODESystem
    ${SRC_DIR}/ODESystem/grimech/out
    ${SRC_DIR}/ODESystem/grimech/out/jacobs
    ${SRC_DIR}/ODESystem/grimech/out/rates
    ${SRC_DIR}/Integrators
    ${INC_DIR}/Integrators
)

target_link_libraries(seulex3_mpi PRIVATE MPI::MPI_CXX)

# Для CUDA-оптимизаций при работе с жёсткими ОДУ
target_compile_options(seulex3_mpi PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math>
)

set_target_properties(seulex3_mpi PROPERTIES
    CUDA_ARCHITECTURES "120"
    CUDA_SEPARABLE_COMPILATION ON
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(seulex3_mpi PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-G -lineinfo>
    )
endif()
