cmake_minimum_required(VERSION 3.15)
project(MyProject VERSION 1.0.0 LANGUAGES CXX)

# Устанавливаем стандарт C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Опционально: включаем кэш для ускорения сборки
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Добавляем папку external как место для поиска CMake-проектов
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/external")

# Подключаем внешние библиотеки
add_subdirectory(external/rapidjson)

# Ваш основной проект
add_subdirectory(src)