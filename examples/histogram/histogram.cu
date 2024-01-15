#include <cuda_runtime.h>
#include <stdio.h>

__global__ void gmem_histo_kernel(unsigned char* random_bytes, int stream_length, int* histo) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; i < stream_length; i += stride) {
        atomicAdd(&(histo[random_bytes[i]]), 1);
    }
}

void cuda_gmem_count_frequency(unsigned char* random_bytes, int stream_length, int* histo) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    unsigned char* dev_random_bytes;
    int* dev_histo;

    cudaMalloc((void**)&dev_random_bytes, stream_length);
    cudaMemcpy(dev_random_bytes, random_bytes, stream_length, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&dev_histo, 256*sizeof(int));
    cudaMemset(dev_histo, 0, 256*sizeof(int));

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int blocks = prop.multiProcessorCount;
    gmem_histo_kernel<<<blocks*2, 256>>>(dev_random_bytes, stream_length, dev_histo);

    cudaMemcpy(histo, dev_histo, 256*sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("Cuda with global memory count time elapsed: %.6f ms\n", elapsed);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(dev_random_bytes);
    cudaFree(dev_histo);
}

__global__ void smem_histo_kernel(unsigned char* random_bytes, int stream_length, int* histo) {
    __shared__ int temp[256];
    temp[threadIdx.x] = 0;
    __syncthreads();

    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; i < stream_length; i += stride) {
        atomicAdd(&(temp[random_bytes[i]]), 1);
    }

    __syncthreads();
    atomicAdd(&(histo[threadIdx.x]), temp[threadIdx.x]);
}

void cuda_smem_count_frequency(unsigned char* random_bytes, int stream_length, int* histo) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    unsigned char* dev_random_bytes;
    int* dev_histo;

    cudaMalloc((void**)&dev_random_bytes, stream_length);
    cudaMemcpy(dev_random_bytes, random_bytes, stream_length, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&dev_histo, 256*sizeof(int));
    cudaMemset(dev_histo, 0, 256*sizeof(int));

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int blocks = prop.multiProcessorCount;
    smem_histo_kernel<<<blocks*2, 256>>>(dev_random_bytes, stream_length, dev_histo);

    cudaMemcpy(histo, dev_histo, 256*sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("Cuda with shared memory count time elapsed: %.6f ms\n", elapsed);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(dev_random_bytes);
    cudaFree(dev_histo);
}