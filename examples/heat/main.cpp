#include <iostream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <stdio.h>
#include <getopt.h>
#include "helper.cuh"

void init_notexture(DataBlock* d);
void destroy_notexture(DataBlock* d);
void draw_notexture(DataBlock* d);

void tex1d_init(DataBlock* d);
void tex1d_destroy(DataBlock* d);
void tex1d_draw(DataBlock* d);

void tex2d_init(DataBlock* d);
void tex2d_destroy(DataBlock* d);
void tex2d_draw(DataBlock* d);

void cuda_draw(int memory_type) {
    if (!glfwInit()) {
        throw std::runtime_error("failed to init");
    }

    GLFWwindow* window = glfwCreateWindow(DIM, DIM, "Texture", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        throw std::runtime_error("failed to create glfw window");
    }

    glfwMakeContextCurrent(window);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        glfwDestroyWindow(window);
        glfwTerminate();
        throw std::runtime_error("failed to init glew");
    }

    GLuint buffer;
    glGenBuffers(1, &buffer);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buffer);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, DIM*DIM*4, nullptr, GL_DYNAMIC_DRAW);

    cudaGraphicsResource *cuda_resource;
    cudaError_t cuda_status = cudaGraphicsGLRegisterBuffer(&cuda_resource, buffer, cudaGraphicsMapFlagsNone);
    if (cuda_status != cudaSuccess) {
        throw std::runtime_error("Failed to register CUDA graphics resource");
    }

    uchar4 *d_out;
    size_t num_bytes;
    cudaGraphicsMapResources(1, &cuda_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void **)&d_out, &num_bytes, cuda_resource);

    void (*init)(DataBlock*) = nullptr;
    void (*draw)(DataBlock*) = nullptr;
    void (*destroy)(DataBlock*) = nullptr;

    if (memory_type == 0) {
        init = init_notexture;
        draw = draw_notexture;
        destroy = destroy_notexture;
        printf("use device memory\n");
    } else if (memory_type == 1) {
        init = tex1d_init;
        draw = tex1d_draw;
        destroy = tex1d_destroy;
        printf("use texture1d memory\n");
    } else {
        init = tex2d_init;
        draw = tex2d_draw;
        destroy = tex2d_destroy;
        printf("use texture2d memory\n");
    }

    DataBlock data;
    init(&data);

    data.bitmap_ptr = d_out;

    while (!glfwWindowShouldClose(window)) {
        draw(&data);
        
        cudaDeviceSynchronize();

        glClear(GL_COLOR_BUFFER_BIT);
        
        glDrawPixels(DIM, DIM, GL_RGBA, GL_UNSIGNED_BYTE, 0);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    cudaGraphicsUnmapResources(1, &cuda_resource, 0);
    cudaGraphicsUnregisterResource(cuda_resource);

    destroy(&data);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    glDeleteBuffers(1, &buffer);
    glfwDestroyWindow(window);
    glfwTerminate();
}

void usage(const char *prog_name) {
    printf("Usage: %s [options]\n", prog_name);
    printf("Program options:\n");
    printf("   -h --help         Show usage\n");
    printf("   -m --memory <0/1/2> Select device memory type: 0 for global and 1 for tex1d and 2 for tex2d (default=0)\n");
}

int main(int argc, char** argv) {
    int opt;
    static struct option long_options[]={
        {"help", 0, 0, 'h'},
        {"memory", 1, 0, 'm'},
        {0, 0, 0, 0}
    };

    int memory_type = 0;
    while ((opt = getopt_long(argc, argv, "m:h", long_options, nullptr)) != EOF) {
        switch (opt) {
        case 'm':
            memory_type = std::atoi(optarg);
            if (memory_type != 0 && memory_type != 1 && memory_type != 2) {
                usage(argv[0]);
                return 1;
            }
            break;
        default:
            usage(argv[0]);
            return 1;
        }
    }

    cuda_draw(memory_type);
}