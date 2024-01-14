#include <cuda_runtime.h>
#include <stdio.h>
#include <random>

#define INF 2e10f
#define SPHERES 20

struct Sphere {
    float r, g, b;
    float radius;
    float x, y, z;
    
    __device__ float hit(float ox, float oy, float* n) {
        float dx = ox - x;
        float dy = oy - y;
        if (dx*dx + dy*dy < radius*radius) {
            float dz = sqrtf(radius*radius - dx*dx - dy*dy);
            *n = dz / sqrtf(radius*radius);
            return dz + z;
        }
        return -INF;
    }
};

__global__ void kernel(Sphere* s, uchar3* ptr, int width, int height) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    float ox = (x - width/2);
    float oy = (y - height/2);

    float r = 0, g = 0, b = 0;
    float maxz = -INF;
    for (int i = 0; i < SPHERES; ++i) {
        float n;
        float t = s[i].hit(ox, oy, &n);    
        if (t > maxz) {
            float fscale = n;
            r = s[i].r * fscale;
            g = s[i].g * fscale;
            b = s[i].b * fscale;
            maxz = t;
        }
    }
       
    ptr[offset] = make_uchar3(r*255, g*255, b*255);
}

Sphere* init_global_sphere() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis_color(0.0f, 1.0f);
    std::uniform_real_distribution<float> dis_pos(-500.0f, 500.0f);
    std::uniform_real_distribution<float> dis_radius(20.0f, 120.0f);

    Sphere *temp_s = (Sphere*)malloc(sizeof(Sphere)*SPHERES);
    Sphere *s;
    cudaMalloc((void**)&s, sizeof(Sphere)*SPHERES);
    for (int i = 0; i < SPHERES; ++i) {
        temp_s[i].r = dis_color(gen);
        temp_s[i].g = dis_color(gen);
        temp_s[i].b = dis_color(gen);
        temp_s[i].x = dis_pos(gen);
        temp_s[i].y = dis_pos(gen);
        temp_s[i].z = dis_pos(gen);
        temp_s[i].radius = dis_radius(gen);
    }
    cudaMemcpy(s, temp_s, sizeof(Sphere)*SPHERES, cudaMemcpyHostToDevice);
    free(temp_s);

    printf("init global sphere success\n");
    return s;
}

void destroy_global_sphere(Sphere *s) {
    cudaFree(s);
    printf("destroy global sphere success\n");
}

void draw_nonst(Sphere* s, uchar3* ptr, int width, int height) {
    dim3 block_dim(16, 16);
    dim3 grid_dim(width/16, height/16);

    kernel<<<grid_dim, block_dim>>>(s, ptr, width, height);
}