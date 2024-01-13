#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <algorithm>

__global__ void kernel(uchar3* ptr) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    __shared__ float shared[16][16];
    const float period = 128.0f;
    
    shared[threadIdx.x][threadIdx.y] = 255 * (sinf(x*2.0f*M_PI/period) + 1.0f) *
                                            (sinf(y*2.0f*M_PI/period) + 1.0f) / 4.0f;
    
    __syncthreads();
    ptr[offset] = make_uchar3(0, shared[15-threadIdx.x][15-threadIdx.y], 0);
}


void draw(uchar3* ptr, int width, int height) {
    dim3 block_dim(16, 16);
    dim3 grid_dim(width/16, height/16);

    kernel<<<grid_dim, block_dim>>>(ptr);
}

void print_device_msg() {
    int count = 0;
    cudaGetDeviceCount(&count);
    printf("device count: %d\n", count);

    cudaDeviceProp prop;
    for (int i = 0; i < count; ++i) {
        cudaGetDeviceProperties(&prop, i);
        printf("-----device: %d\n", i);
        printf("设备名: %s\n", prop.name);
        printf("SM数量: %d\n", prop.multiProcessorCount);
        printf("每个线程块的共享内存大小：%fKB\n", prop.sharedMemPerBlock / 1024.0);
        printf("每个线程块的最大线程数: %d\n", prop.maxThreadsPerBlock);
        printf("每个block的最大dim: [%d, %d, %d]\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("每个grid的最大dim: [%d, %d, %d]\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("每个SM的最大线程数: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("每个SM的最大线程束数: %d\n", prop.maxThreadsPerMultiProcessor / 32);
    }
}
