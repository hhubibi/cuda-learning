#include <iostream>
#include <complex>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <random>
#include <stdio.h>
#include <getopt.h>

static constexpr int max_iter = 200;

int width = 600;
int height = 600;
float scale = 1.5;
float c_x = 0.355;
float c_y = 0.355;

void cuda_draw_julia(uchar3* ptr, int width, int height, float scale, int max_iter, float c_x, float c_y);

void draw_julia(int width, int height) {
    glLoadIdentity();
    glOrtho(0, width, 0, height, -1, 1);

    glBegin(GL_POINTS);

    for (int x = 0; x < width; ++x) {
        for (int y = 0; y < height; ++y) {
            std::complex<float> a(scale * (float)(width/2 - x) / (width/2), scale * (float)(height/2 - y) / (height/2));
            std::complex<float> c(c_x, c_y);

            int i = 0;
            for (; i < max_iter; ++i) {
                a = a * a + c;
                if (abs(a) > 10) {
                    break;
                }
            }

            int flag = (i == max_iter) ? 1 : 0;
            glColor3f(255 * flag, 0, 0);
            glVertex2i(x, y);
        }
    }

    glEnd();
}

void cpu_draw() {
    if (!glfwInit()) {
        throw std::runtime_error("failed to init");
    }

    GLFWwindow* window = glfwCreateWindow(width, height, "Julia Set", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        throw std::runtime_error("failed to create window");
    }

    glfwMakeContextCurrent(window);
    double last_time = glfwGetTime();
    int frame_count = 0;

    while (!glfwWindowShouldClose(window)) {
        double current_time = glfwGetTime();
        frame_count++;

        if (current_time - last_time >= 1.0) {
            std::cout << "FPS: " << frame_count << std::endl;
            frame_count = 0;
            last_time += 1.0;
        }

        glClear(GL_COLOR_BUFFER_BIT);

        draw_julia(width, height);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwDestroyWindow(window);
    glfwTerminate();
}

void cuda_draw() {
    if (!glfwInit()) {
        throw std::runtime_error("failed to init");
    }

    GLFWwindow* window = glfwCreateWindow(width, height, "Julia Set", nullptr, nullptr);
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

    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, width, height, 0, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    cudaGraphicsResource *cuda_resource;
    cudaError_t cuda_status = cudaGraphicsGLRegisterBuffer(&cuda_resource, buffer, cudaGraphicsMapFlagsNone);
    if (cuda_status != cudaSuccess) {
        throw std::runtime_error("Failed to register CUDA graphics resource");
    }

    double last_time = glfwGetTime();
    int frame_count = 0;
    while (!glfwWindowShouldClose(window)) {
        double current_time = glfwGetTime();
        frame_count++;

        if (current_time - last_time >= 1.0) {
            std::cout << "FPS: " << frame_count << std::endl;
            frame_count = 0;
            last_time += 1.0;
        }

        uchar3 *d_out;
        size_t num_bytes;
        cudaGraphicsMapResources(1, &cuda_resource, 0);
        cudaGraphicsResourceGetMappedPointer((void **)&d_out, &num_bytes, cuda_resource);

        cuda_draw_julia(d_out, width, height, scale, max_iter, c_x, c_y);
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

void usage(const char* prog_name) {
    printf("Usage: %s [options]\n", prog_name);
    printf("Program options:\n");
    printf("   -r --render <cpu/gpu> Select render: cpu or gpu (default=cpu)\n");
    printf("   -w --width            Set window width (default=512)\n");
    printf("   -h --height           Set window height (default=512)\n");
    printf("   -s --scale            Set scale param (default=1.5)\n");
    printf("   --cx                  Set c real (default=0.355)\n");
    printf("   --cy                  Set c virtual (default=0.355)\n");
    printf("   -? --help             This message\n");
}

int main(int argc, char** argv) {
    // std::default_random_engine generator;
    // std::uniform_real_distribution<float> distribution(-1.0, 1.0);

    // c_x = distribution(generator);
    // c_y = distribution(generator);

    int opt;
    static struct option long_options[]={
        {"help", 0, 0, 'h'},
        {"render", 1, 0, 'r'},
        {"width", 1, 0, 'w'},
        {"height", 1, 0, 'h'},
        {"scale", 1, 0, 's'},
        {"cx", 1, 0, 256},
        {"cy", 1, 0, 257},
        {0, 0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, "w:h:s:r:?", long_options, nullptr)) != EOF) {
        switch (opt) {
        case 'r':
            if (std::string(optarg).compare("gpu") == 0) {
                cuda_draw();   
            } else {
                cpu_draw();
            }
            break;
        case 'h':
            height = std::atoi(optarg);
            break;
        case 'w':
            width = std::atoi(optarg);
            break;
        case 's':
            scale = std::atof(optarg);
            break;
        case 256:
            c_x = std::atof(optarg);
            break;
        case 257:
            c_y = std::atof(optarg);
            break;
        case '?':
        default:
            usage(argv[0]);
            return 1;
        }
    }

    return 0;
}

