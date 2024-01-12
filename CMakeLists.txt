cmake_minimum_required(VERSION 3.8)

project(cuda-example LANGUAGES C CXX CUDA)

function(build_example example_name)
    set(example_folder ${CMAKE_CURRENT_SOURCE_DIR}/${example_name})
    message(STATUS "Generating project file for example in ${example_folder}")

    file(GLOB cuda_files ${example_folder}/*.cu)
	set(main_cpp ${example_folder}/main.cpp)

    add_executable(${example_name} ${cuda_files} ${main_cpp})
    
    if(${example_name} STREQUAL "add")
        find_package(OpenMP REQUIRED)
        if(OpenMP_CXX_FOUND)
            target_link_libraries(${example_name} OpenMP::OpenMP_CXX)
        endif()
    endif()
endfunction(build_example)


function(build_examples)
    foreach(example ${examples})
        build_example(${example})
    endforeach(example)
endfunction(build_examples)


set(examples
    add
    # julia
    )

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_CXX_STANDARD 17)

build_examples()
