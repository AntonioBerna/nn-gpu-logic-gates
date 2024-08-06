cmake_minimum_required(VERSION 3.28)

project(nn-gnu-logic-gates LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 61)

enable_language(CUDA)

add_executable(nn-gnu-logic-gates src/main.cu)
