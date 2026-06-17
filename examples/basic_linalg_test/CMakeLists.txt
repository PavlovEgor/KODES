cmake_minimum_required(VERSION 3.15)
project(ConfigExample LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Для CUDA
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo -Xptxas=-v")

# Добавьте флаг для агрессивного инлайнинга
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas=-allow-expensive-optimizations=true")

# Исправлено: использование ${} для подстановки переменных
set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../)
set(SRC_DIR ${ROOT_DIR}/src)
set(INC_DIR ${ROOT_DIR}/include)

# Убираем предупреждение о CUDA архитектуре
set(CMAKE_CUDA_ARCHITECTURES "120")  # для RTX 20xx, 30xx, 40xx, 50xx

# Исправлено: правильный синтаксис для путей к файлам
add_executable( basic_linalg_test
    basic_linalg_test.cu
    ${SRC_DIR}/basic_linalg.cu
)

target_include_directories(basic_linalg_test PRIVATE
    ${INC_DIR}
    ${INC_DIR}/integrators
    ${ROOT_DIR}/external/rapidjson/include
    ${ROOT_DIR}/examples/TestSystems
)

# Дополнительные рекомендации:

# 1. Для CUDA-оптимизаций при работе с жёсткими ОДУ
target_compile_options(basic_linalg_test PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math>
)

# 2. Если используете double precision (рекомендуется для этой задачи)
set_target_properties(basic_linalg_test PROPERTIES
    CUDA_ARCHITECTURES "120"
    CUDA_SEPARABLE_COMPILATION ON
)

# 3. Для отладки (опционально)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(basic_linalg_test PRIVATE 
        $<$<COMPILE_LANGUAGE:CUDA>:-G -lineinfo>
    )
endif()