#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>

#include "helper.cuh"

#include <cuda_texture_types.h>


__global__ void tex1d_copy_const_kernel(float* iptr, cudaTextureObject_t tex) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int offset = x + y * blockDim.x * gridDim.x;

    float c = c = tex1Dfetch<float>(tex, offset);
    if (c != 0) iptr[offset] = c;
}

__global__ void tex1d_blend_kernel(float* dst, cudaTextureObject_t tex) {
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

    float t, l, c, r, b;
    t = tex1Dfetch<float>(tex, top);
    l = tex1Dfetch<float>(tex, left);
    c = tex1Dfetch<float>(tex, offset);
    r = tex1Dfetch<float>(tex, right);
    b = tex1Dfetch<float>(tex, bottom);

    dst[offset] = c + SPEED * (t + b + r + l - c*4);
}

void create_tex_obj(cudaTextureObject_t *tex, float* ptr, int size_in_bytes) {
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = ptr;
    resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
    resDesc.res.linear.desc.x = 32; // bits per channel
    resDesc.res.linear.sizeInBytes = size_in_bytes;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;

    // create texture object: we only have to do this once!
    cudaCreateTextureObject(tex, &resDesc, &texDesc, nullptr);
}

void tex1d_init(DataBlock* d) {
    cudaEventCreate(&d->start);
    cudaEventCreate(&d->end);

    int image_size = DIM*DIM*4;

    cudaMalloc((void**)&d->in_ptr, image_size);
    cudaMalloc((void**)&d->out_ptr, image_size);
    cudaMalloc((void**)&d->const_ptr, image_size);

    create_tex_obj(&d->tex_const, d->const_ptr, image_size);
    create_tex_obj(&d->tex_in, d->in_ptr, image_size);
    create_tex_obj(&d->tex_out, d->out_ptr, image_size);

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

void tex1d_destroy(DataBlock* d) {
    cudaDestroyTextureObject(d->tex_const);
    cudaDestroyTextureObject(d->tex_in);
    cudaDestroyTextureObject(d->tex_out);

    cudaFree(d->in_ptr);
    cudaFree(d->out_ptr);
    cudaFree(d->const_ptr);

    cudaEventDestroy(d->start);
    cudaEventDestroy(d->end);

    printf("destroy success\n");
}


void tex1d_draw(DataBlock* d) {
    cudaEventRecord(d->start);

    dim3 block_dim(16, 16);
    dim3 grid_dim(DIM/16, DIM/16);

    volatile bool dst_out = true;

    for (int i = 0; i < 90; ++i) {
        float *in, *out;
        cudaTextureObject_t tex;
        if (dst_out) {
            in = d->in_ptr;
            out = d->out_ptr;
            tex = d->tex_out;
        } else {
            out = d->in_ptr;
            in = d->out_ptr;
            tex = d->tex_in;
        }
        tex1d_copy_const_kernel<<<grid_dim, block_dim>>>(in, d->tex_const);
        tex1d_blend_kernel<<<grid_dim, block_dim>>>(out, tex);
        dst_out = !dst_out;
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