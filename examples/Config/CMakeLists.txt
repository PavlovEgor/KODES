cmake_minimum_required(VERSION 3.15)
project(ConfigExample LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Убираем предупреждение о CUDA архитектуре
set(CMAKE_CUDA_ARCHITECTURES "120")  # для RTX 20xx, 30xx, 40xx
# Или можно указать несколько архитектур:
# set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86")

add_executable(config_example test.cu ../../src/kodes_config.cu)

target_include_directories(config_example PRIVATE
    ../../include
    ../../external/rapidjson/include
)