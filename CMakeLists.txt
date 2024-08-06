cmake_minimum_required(VERSION 3.28)

project(nn-gpu-logic-gates LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 61)

enable_language(CUDA)

add_executable(nn-gpu-logic-gates src/main.cu)
