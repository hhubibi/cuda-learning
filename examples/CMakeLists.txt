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

    if(${example_name} STREQUAL "julia")
        find_package(OpenGL REQUIRED)
        if(OPENGL_FOUND)
            target_link_libraries(${example_name} glfw GLEW OpenGL::GL)
        endif()
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
    )

build_examples()
