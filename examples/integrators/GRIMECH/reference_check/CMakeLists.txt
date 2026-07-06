cmake_minimum_required(VERSION 3.15)
project(PyjacProbe LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../)
set(SRC_DIR ${ROOT_DIR}/src)

set(CMAKE_CUDA_ARCHITECTURES "120")

# Никакого KODES/Seulex здесь не нужно - только сырые функции, которые
# сгенерировал pyJac, ровно тот же набор .cu, что собирается в seulex3.
add_executable(pyjac_probe
    pyjac_probe.cu
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

target_include_directories(pyjac_probe PRIVATE
    ${SRC_DIR}/ODESystem/grimech/out
    ${SRC_DIR}/ODESystem/grimech/out/jacobs
    ${SRC_DIR}/ODESystem/grimech/out/rates
)

target_compile_options(pyjac_probe PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

set_target_properties(pyjac_probe PROPERTIES
    CUDA_ARCHITECTURES "120"
    CUDA_SEPARABLE_COMPILATION ON
)

# state.txt читается по относительному пути - копируем его рядом с бинарником,
# чтобы `./pyjac_probe` без аргументов сразу находил его в build/.
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/state.txt ${CMAKE_CURRENT_BINARY_DIR}/state.txt COPYONLY)
