cmake_minimum_required(VERSION 3.29)
project(minecraft-weekend LANGUAGES C CUDA)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_C_STANDARD 11)

add_executable(minecraft-weekend
        src/main.c
)

set_source_files_properties(
        src/main.c
        PROPERTIES
        LANGUAGE C
)

set_target_properties(minecraft-weekend PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
