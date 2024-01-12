cmake_minimum_required(VERSION 3.0.0)
project(hello-world VERSION 0.1.0 LANGUAGES C CXX CUDA)


find_package(OpenMP REQUIRED)
file(GLOB cuda_files src/add.cu)

add_executable(${PROJECT_NAME} main.cpp ${cuda_files})

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
endif()
