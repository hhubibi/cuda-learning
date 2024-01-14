#include <iostream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <stdio.h>
#include <getopt.h>

static constexpr int width = 1024;
static constexpr int height = 1024;

struct Sphere;

// const memory in device
void draw(uchar3* ptr, int width, int height);
void init_const_sphere();
void destroy_const_sphere();

// global memory in device
void draw_nonst(Sphere* s, uchar3* ptr, int width, int height);
Sphere* init_global_sphere();
void destroy_global_sphere(Sphere* s);

void cuda_draw(int use_const) {
    if (!glfwInit()) {
        throw std::runtime_error("failed to init");
    }

    GLFWwindow* window = glfwCreateWindow(width, height, "Ray Tracing", nullptr, nullptr);
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
    glBufferData(GL_PIXEL_UNPACK_BUFFER, width*height*3, nullptr, GL_DYNAMIC_DRAW);

    cudaGraphicsResource *cuda_resource;
    cudaError_t cuda_status = cudaGraphicsGLRegisterBuffer(&cuda_resource, buffer, cudaGraphicsMapFlagsNone);
    if (cuda_status != cudaSuccess) {
        throw std::runtime_error("Failed to register CUDA graphics resource");
    }

    uchar3 *d_out;
    size_t num_bytes;
    cudaGraphicsMapResources(1, &cuda_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void **)&d_out, &num_bytes, cuda_resource);

    Sphere* s;

    if (use_const) {
        init_const_sphere();
    } else {
        s = init_global_sphere();
    }

    double last_time = glfwGetTime();
    int frame_count = 0;
    int ticks = 0;
    while (!glfwWindowShouldClose(window)) {
        double current_time = glfwGetTime();
        frame_count++;

        if (current_time - last_time >= 1.0) {
            std::cout << "FPS: " << frame_count << std::endl;
            frame_count = 0;
            last_time += 1.0;
        }
        ticks++;

        if (use_const) {
            draw(d_out, width, height);
        } else {
            draw_nonst(s, d_out, width, height);
        }
        
        cudaDeviceSynchronize();

        glClear(GL_COLOR_BUFFER_BIT);
        
        glDrawPixels(width, height, GL_RGB, GL_UNSIGNED_BYTE, 0);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    cudaGraphicsUnmapResources(1, &cuda_resource, 0);
    cudaGraphicsUnregisterResource(cuda_resource);

    if (use_const) {
        destroy_const_sphere();
    } else {
        destroy_global_sphere(s);
    }

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    glDeleteBuffers(1, &buffer);
    glfwDestroyWindow(window);
    glfwTerminate();
}

void usage(const char *prog_name) {
    printf("Usage: %s [options]\n", prog_name);
    printf("Program options:\n");
    printf("   -h --help         Show usage\n");
    printf("   -m --memory <0/1> Select device memory type: 0 for const and 1 for global (default=0)\n");
}

int main(int argc, char** argv) {
    int opt;
    static struct option long_options[]={
        {"help", 0, 0, 'h'},
        {"memory", 1, 0, 'm'},
        {0, 0, 0, 0}
    };

    int use_const = 0;
    while ((opt = getopt_long(argc, argv, "m:h", long_options, nullptr)) != EOF) {
        switch (opt) {
        case 'm':
            use_const = std::atoi(optarg);
            break;
        default:
            usage(argv[0]);
            return 1;
        }
    }

    cuda_draw(use_const);
}