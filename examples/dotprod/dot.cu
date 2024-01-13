#include <cuda_runtime.h>
#include <math.h>
#include <iostream>

// 9*1024 is too large, error result
#define N (8*1024)
#define BLOCK_DIM 256

__global__ void dot(float* x, float* y, float* z) {
    __shared__ float cache[BLOCK_DIM];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cid = threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    float temp = 0;
    for (int i = tid; i < N; i += stride) {
        temp += x[tid] * y[tid];
    }
    cache[cid] = temp;

    __syncthreads();

    // reduction
    int i = blockDim.x / 2;
    while (i != 0) {
        if (cid < i) {
            cache[cid] += cache[cid + i];
        }
        __syncthreads();
        i /= 2;
    }

    if (cid == 0) {
        z[blockIdx.x] = cache[0];
    }
}

void cuda_dot() {
    int grid_dim = std::min(32, (N + BLOCK_DIM - 1) / BLOCK_DIM);

    float *x, *y, *z;

    cudaMallocManaged((void**)&x, sizeof(float)*N);
    cudaMallocManaged((void**)&y, sizeof(float)*N);
    cudaMallocManaged((void**)&z, sizeof(float)*grid_dim);

    for (int i = 0; i < N; ++i) {
        x[i] = i;
        y[i] = 2*i;
    }

    dot<<<grid_dim, BLOCK_DIM>>>(x, y, z);
    cudaDeviceSynchronize();

    float sum = 0;
    for (int i = 0; i < grid_dim; ++i) {
        sum += z[i];
    }

    #define sum_squares(a) (a*(a+1)*(2*a+1)/6)
    printf("GPU value: %.6f; real value: %.6f\n", sum, 2*sum_squares((float)(N-1)));

    cudaFree(x);
    cudaFree(y);
    cudaFree(z);
}