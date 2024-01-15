function(build_example example_name)
    set(example_folder ${CMAKE_CURRENT_SOURCE_DIR}/${example_name})
    message(STATUS "Generating project file for example in ${example_folder}")

    file(GLOB cuda_header_files ${example_folder}/*.cuh)
    file(GLOB cuda_files ${example_folder}/*.cu)
    file(GLOB header_files ${example_folder}/*.h)
	set(main_cpp ${example_folder}/main.cpp)

    add_executable(${example_name} ${header_files} ${cuda_header_files} ${cuda_files} ${main_cpp})
    
    if(${example_name} STREQUAL "add")
        find_package(OpenMP REQUIRED)
        target_link_libraries(${example_name} OpenMP::OpenMP_CXX)
    endif()

    set(ui_examples julia ball raytracing heat)
    
    if(${example_name} IN_LIST ui_examples)
        find_package(GLEW REQUIRED)
        target_link_libraries(${example_name} GLEW::glew)

        find_package(OpenGL REQUIRED)
        target_link_libraries(${example_name} glfw OpenGL::GL)

        find_package(CUDAToolkit REQUIRED)
        target_link_libraries(${example_name} CUDA::cudart)
    endif()
endfunction(build_example)


function(build_examples)
    foreach(example ${examples})
        build_example(${example})
    endforeach(example)
endfunction(build_examples)


set(examples
    add
    julia
    dotprod
    ball
    raytracing
    heat
    histogram
    cuda_stream
    )

build_examples()
