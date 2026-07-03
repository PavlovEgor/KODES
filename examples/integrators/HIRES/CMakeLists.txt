cmake_minimum_required(VERSION 3.15)
project(ConfigExample LANGUAGES CXX)  # Убрали CUDA

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Пути к директориям
set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../)
set(SRC_DIR ${ROOT_DIR}/src)
set(INC_DIR ${ROOT_DIR}/include)

# Теперь используем расширение .cpp вместо .cu
add_executable(hires_example 
    test.cpp  # переименуйте test.cu в test.cpp
    ${SRC_DIR}/kodes_config.cpp  # переименуйте .cu в .cpp
    ${SRC_DIR}/integrators/integrator.cpp
    ${SRC_DIR}/integrators/seulex.cpp
    ${SRC_DIR}/integrators/euler.cpp
    ${ROOT_DIR}/examples/TestSystems/HIRESSystem.cpp
)

target_include_directories(hires_example PRIVATE
    ${INC_DIR}
    ${INC_DIR}/integrators
    ${ROOT_DIR}/external/rapidjson/include
    ${ROOT_DIR}/examples/TestSystems
)

# Настройки для g++
target_compile_options(hires_example PRIVATE
    -O3                    # Оптимизация
    -Wall -Wextra         # Предупреждения
    -march=native         # Оптимизация под вашу архитектуру
    -ffast-math           # Аналог CUDA -use_fast_math
)

# Для отладки
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(hires_example PRIVATE
        -g -O0 -ggdb3      # Отладочная информация
        -fsanitize=address # Детектор утечек памяти (опционально)
    )
    target_link_options(hires_example PRIVATE
        -fsanitize=address
    )
endif()

# Для релиза
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(hires_example PRIVATE
        -O3 -DNDEBUG
        -flto              # Link-time optimization
    )
endif()

# Связывание с математической библиотекой
target_link_libraries(hires_example PRIVATE
    m  # математическая библиотека
    pthread  # если используете многопоточность
)