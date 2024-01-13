#include <cuda_runtime.h>
#include <stdio.h>
#include <omp.h>
#include <time.h>

#define BLOCK_DIM 512
#define N (1 << 24)

__global__ void add(int* x, int* y, int* z, int n) {
    // eg. grid dim: 6; block dim: 256; threads: 6 * 256
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride) {
        z[i] = x[i] + y[i];
    }
}

float cpu_add(bool use_omp) {
    printf("\n===cpu add begin===\n");
    int* x, *y, *z;

    x = (int*)malloc(sizeof(int)*N);
    y = (int*)malloc(sizeof(int)*N);
    z = (int*)malloc(sizeof(int)*N);

    for (int i = 0; i < N; ++i) {
        x[i] = i;
        y[i] = -i;
    }

    clock_t start = clock();

    if (!use_omp) {
        // raw version
        for (int i = 0; i < N; ++i) {
            z[i] = x[i] + y[i];
        }
    } else {
        // openmp version
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < N; ++i) {
            z[i] = x[i] + y[i];
        }
    }

    clock_t end = clock();

    double elapsed = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;

    printf("time elapsed: %f milliseconds\n", elapsed);

    free(x);
    free(y);
    free(z);

    printf("===cpu add end===\n");
    return elapsed;
}

float cuda_add(int block_dim) {
    printf("\n===cuda add begin===\n");
    int* x, *y, *z;

    x = (int*)malloc(sizeof(int)*N);
    y = (int*)malloc(sizeof(int)*N);
    z = (int*)malloc(sizeof(int)*N);

    for (int i = 0; i < N; ++i) {
        x[i] = i;
        y[i] = -i;
    }

    int* d_x, *d_y, *d_z;
    cudaMalloc((void**)&d_x, sizeof(int)*N);
    cudaMalloc((void**)&d_y, sizeof(int)*N);
    cudaMalloc((void**)&d_z, sizeof(int)*N);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    int grid_dim = (block_dim + N - 1) / block_dim;

    cudaEventRecord(start);

    cudaMemcpy(d_x, x, sizeof(int)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, sizeof(int)*N, cudaMemcpyHostToDevice);

    add<<<grid_dim, block_dim>>>(d_x, d_y, d_z, N);

    cudaMemcpy(z, d_z, sizeof(int)*N,cudaMemcpyDeviceToHost);

    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float elapsed = 0;
    cudaEventElapsedTime(&elapsed, start, end);

    printf("block dim: %d\ntime elapsed: %f milliseconds\n", block_dim, elapsed);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);

    free(x);
    free(y);
    free(z);

    printf("===cuda add end===\n");

    return elapsed;
}


float cuda_add_um(int block_dim) {
    printf("\n===cuda add with unified memory begin===\n");
    int* z;
    z = (int*)malloc(sizeof(int)*N);

    int* d_x, *d_y, *d_z;
    cudaMallocManaged((void**)&d_x, sizeof(int)*N);
    cudaMallocManaged((void**)&d_y, sizeof(int)*N);
    cudaMallocManaged((void**)&d_z, sizeof(int)*N);

    for (int i = 0; i < N; ++i) {
        d_x[i] = i;
        d_y[i] = -i;
    }

    int grid_dim = (block_dim + N - 1) / block_dim;

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);

    add<<<grid_dim, block_dim>>>(d_x, d_y, d_z, N);
    // cudaDeviceSynchronize();
    // memcpy(z, d_z, sizeof(int)*N);

    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float elapsed = 0;
    cudaEventElapsedTime(&elapsed, start, end);

    printf("block dim: %d\ntime elapsed: %f milliseconds\n", block_dim, elapsed);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);

    free(z);

    printf("===cuda add with unified memory end===\n");
    return elapsed;
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
        printf("每个SM的最大线程数: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("每个SM的最大线程束数: %d\n", prop.maxThreadsPerMultiProcessor / 32);
    }
}
