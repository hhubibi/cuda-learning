#include <iostream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <stdio.h>

static constexpr int width = 1024;
static constexpr int height = 1024;

void draw(uchar3* ptr, int width, int height);
void print_device_msg();

void cuda_draw() {
    if (!glfwInit()) {
        throw std::runtime_error("failed to init");
    }

    GLFWwindow* window = glfwCreateWindow(width, height, "Ball", nullptr, nullptr);
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

    double last_time = glfwGetTime();
    int frame_count = 0;
    int ticks = 0;
    while (!glfwWindowShouldClose(window)) {
        double current_time = glfwGetTime();
        frame_count++;

        if (current_time - last_time >= 1.0) {
            // std::cout << "FPS: " << frame_count << std::endl;
            frame_count = 0;
            last_time += 1.0;
        }
        ticks++;

        uchar3 *d_out;
        size_t num_bytes;
        cudaGraphicsMapResources(1, &cuda_resource, 0);
        cudaGraphicsResourceGetMappedPointer((void **)&d_out, &num_bytes, cuda_resource);

        draw(d_out, width, height);
        cudaDeviceSynchronize();

        cudaGraphicsUnmapResources(1, &cuda_resource, 0);
  
        glClear(GL_COLOR_BUFFER_BIT);
        
        glDrawPixels(width, height, GL_RGB, GL_UNSIGNED_BYTE, 0);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    cudaGraphicsUnregisterResource(cuda_resource);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    glDeleteBuffers(1, &buffer);
    glfwDestroyWindow(window);
    glfwTerminate();
}


int main() {
    print_device_msg();
    cuda_draw();
}