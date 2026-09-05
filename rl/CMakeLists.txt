cmake_minimum_required(VERSION 3.24)
project(astroswarm_rl_native LANGUAGES CXX)

option(ASTRO_RL_NATIVE_ARCH "Optimize for the build host CPU" ON)
option(ASTRO_RL_FAST_MATH "Enable relaxed floating-point optimizations" ON)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenMP REQUIRED)

pybind11_add_module(_native MODULE
    cpp/bindings.cpp
    cpp/simulator.cpp
)
target_include_directories(_native PRIVATE cpp)
target_compile_features(_native PRIVATE cxx_std_20)
target_link_libraries(_native PRIVATE OpenMP::OpenMP_CXX)

if(MSVC)
    target_compile_options(_native PRIVATE /O2)
else()
    target_compile_options(_native PRIVATE -O3 -Wall -Wextra -Wpedantic)
    if(ASTRO_RL_NATIVE_ARCH)
        target_compile_options(_native PRIVATE -march=native)
    endif()
    if(ASTRO_RL_FAST_MATH)
        target_compile_options(_native PRIVATE -ffast-math)
    endif()
endif()

install(TARGETS _native LIBRARY DESTINATION astroswarm_rl)
