cmake_minimum_required(VERSION 3.15)
project(ConfigExample LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Исправлено: использование ${} для подстановки переменных
set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../)
set(SRC_DIR ${ROOT_DIR}/src)
set(INC_DIR ${ROOT_DIR}/include)

# Убираем предупреждение о CUDA архитектуре
set(CMAKE_CUDA_ARCHITECTURES "120")  # для RTX 20xx, 30xx, 40xx, 50xx

# Исправлено: правильный синтаксис для путей к файлам
add_executable( seulex4
    seulex4.cu
    ${SRC_DIR}/basic_linalg.cu
    ${SRC_DIR}/Resources/SeulexDeviceResources.cu
    ${SRC_DIR}/Resources/DeviceResources.cu
    ${SRC_DIR}/ODESystem/HIRESSystem.cu
    # ${SRC_DIR}/Resources/Operator.cu
    ${SRC_DIR}/Resources/HostResources.cu

    )


target_include_directories(seulex4 PRIVATE
    ${INC_DIR}
    ${INC_DIR}/Resources
    ${INC_DIR}/ODESystem
    # ${ROOT_DIR}/external/rapidjson/include
    # ${ROOT_DIR}/examples/TestSystems
)

# Дополнительные рекомендации:

# 1. Для CUDA-оптимизаций при работе с жёсткими ОДУ
target_compile_options(seulex4 PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math>
)

# 2. Если используете double precision (рекомендуется для этой задачи)
set_target_properties(seulex4 PROPERTIES
    CUDA_ARCHITECTURES "120"
    CUDA_SEPARABLE_COMPILATION ON
)

# 3. Для отладки (опционально)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(seulex4 PRIVATE 
        $<$<COMPILE_LANGUAGE:CUDA>:-G -lineinfo>
    )
endif()