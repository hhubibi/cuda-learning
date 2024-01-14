#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>

#include "helper.cuh"

__global__ void copy_const_kernel(float* iptr, const float* cptr) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int offset = x + y * blockDim.x * gridDim.x;

    if (cptr[offset] != 0) iptr[offset] = cptr[offset];
}

__global__ void blend_kernel(float* out, const float* in) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int offset = x + y * blockDim.x * gridDim.x;

    int left = offset -1;
    int right = offset + 1;
    if (x == 0) left++;
    if (x == DIM - 1) right--;

    int top = offset - DIM;
    int bottom = offset + DIM;
    if (y == 0) top += DIM;
    if (y == DIM - 1) bottom -= DIM;

    out[offset] = in[offset] + SPEED * (in[top] + in[bottom] + in[left] + in[right] - in[offset]*4);
}

void init_notexture(DataBlock* d) {
    cudaEventCreate(&d->start);
    cudaEventCreate(&d->end);

    int image_size = DIM*DIM*4;

    cudaMalloc((void**)&d->in_ptr, image_size);
    cudaMalloc((void**)&d->out_ptr, image_size);
    cudaMalloc((void**)&d->const_ptr, image_size);

    float *temp = (float*)malloc( image_size );
    for (int i = 0; i < DIM*DIM; ++i) {
        temp[i] = 0;
        int x = i % DIM;
        int y = i / DIM;
        if ((x > 300) && (x < 600) && (y > 310) && (y < 601))
            temp[i] = MAX_TEMP;
    }
    temp[DIM*100 + 100] = (MAX_TEMP + MIN_TEMP)/2;
    temp[DIM*700 + 100] = MIN_TEMP;
    temp[DIM*300 + 300] = MIN_TEMP;
    temp[DIM*200 + 700] = MIN_TEMP;
    for (int y = 800; y < 900; y++) {
        for (int x = 400; x < 500; x++) {
            temp[x + y*DIM] = MIN_TEMP;
        }
    }
    cudaMemcpy(d->const_ptr, temp, image_size, cudaMemcpyHostToDevice);

    for (int y = 800; y < DIM; y++) {
        for (int x = 0; x < 200; x++) {
            temp[x + y*DIM] = MAX_TEMP;
        }
    }
    cudaMemcpy(d->in_ptr, temp, image_size, cudaMemcpyHostToDevice);

    free(temp);

    printf("init success\n");
}

void destroy_notexture(DataBlock* d) {
    cudaFree(d->in_ptr);
    cudaFree(d->out_ptr);
    cudaFree(d->const_ptr);

    cudaEventDestroy(d->start);
    cudaEventDestroy(d->end);

    printf("destroy success\n");
}


void draw_notexture(DataBlock* d) {
    cudaEventRecord(d->start);

    dim3 block_dim(16, 16);
    dim3 grid_dim(DIM/16, DIM/16);

    for (int i = 0; i < 90; ++i) {
        copy_const_kernel<<<grid_dim, block_dim>>>(d->in_ptr, d->const_ptr);
        blend_kernel<<<grid_dim, block_dim>>>(d->out_ptr, d->in_ptr);
        std::swap(d->in_ptr, d->out_ptr);
    }

    float_to_color<<<grid_dim, block_dim>>>(d->bitmap_ptr, d->in_ptr);

    cudaEventRecord(d->end);
    cudaEventSynchronize(d->end);

    float elapsed;
    cudaEventElapsedTime(&elapsed, d->start, d->end);
    d->total_time += elapsed;
    ++d->frames;

    printf("Average time per frame: %.f ms\n", d->total_time / d->frames);
}