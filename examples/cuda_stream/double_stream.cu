#include <cuda_runtime.h>
#include <stdio.h>

__global__ void double_stream_kernel(int N, int *a, int *b, int *c) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        int idx1 = (idx + 1) % 256;
        int idx2 = (idx + 2) % 256;
        float   as = (a[idx] + a[idx1] + a[idx2]) / 3.0f;
        float   bs = (b[idx] + b[idx1] + b[idx2]) / 3.0f;
        c[idx] = (as + bs) / 2;
    }
}

void double_stream() {
    int N = 1024*1024;
    int FULL_DATA_SIZE = N*20;

    cudaDeviceProp prop;
    int device_id;
    cudaGetDevice(&device_id);
    cudaGetDeviceProperties(&prop, device_id);
    if (!prop.deviceOverlap) {
        printf("Device will not handle overlaps, so no speed up from streams\n");
        return;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaStream_t stream0, stream1;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);

    int *host_a, *host_b, *host_c;
    int *dev_a0, *dev_b0, *dev_c0;
    int *dev_a1, *dev_b1, *dev_c1;

    cudaMalloc((void**)&dev_a0, N*sizeof(int));
    cudaMalloc((void**)&dev_b0, N*sizeof(int));
    cudaMalloc((void**)&dev_c0, N*sizeof(int));
    cudaMalloc((void**)&dev_a1, N*sizeof(int));
    cudaMalloc((void**)&dev_b1, N*sizeof(int));
    cudaMalloc((void**)&dev_c1, N*sizeof(int));

    cudaHostAlloc((void**)&host_a, FULL_DATA_SIZE*sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void**)&host_b, FULL_DATA_SIZE*sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void**)&host_c, FULL_DATA_SIZE*sizeof(int), cudaHostAllocDefault);

    for (int i = 0; i < FULL_DATA_SIZE; ++i) {
        host_a[i] = rand();
        host_b[i] = rand();
    }

    cudaEventRecord(start, 0);

    for (int i = 0; i < FULL_DATA_SIZE; i += N*2) {
        cudaMemcpyAsync(dev_a0, host_a+i, N*sizeof(int), cudaMemcpyHostToDevice, stream0);
        cudaMemcpyAsync(dev_b0, host_b+i, N*sizeof(int), cudaMemcpyHostToDevice, stream0);
        double_stream_kernel<<<N/256, 256, 0, stream0>>>(N, dev_a0, dev_b0, dev_c0);
        cudaMemcpyAsync(host_c+i, dev_c0, N*sizeof(int), cudaMemcpyDeviceToHost, stream0);

        cudaMemcpyAsync(dev_a1, host_a+i+N, N*sizeof(int), cudaMemcpyHostToDevice, stream1);
        cudaMemcpyAsync(dev_b1, host_b+i+N, N*sizeof(int), cudaMemcpyHostToDevice, stream1);
        double_stream_kernel<<<N/256, 256, 0, stream1>>>(N, dev_a1, dev_b1, dev_c1);
        cudaMemcpyAsync(host_c+i+N, dev_c1, N*sizeof(int), cudaMemcpyDeviceToHost, stream1);
    }
    cudaStreamSynchronize(stream0);
    cudaStreamSynchronize(stream1);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);
    printf( "Time taken with double stream:  %3.1f ms\n", elapsed);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(dev_a0);
    cudaFree(dev_b0);
    cudaFree(dev_c0);
    cudaFree(dev_a1);
    cudaFree(dev_b1);
    cudaFree(dev_c1);

    cudaFreeHost(host_a);
    cudaFreeHost(host_b);
    cudaFreeHost(host_c);
    
    cudaStreamDestroy(stream0);
    cudaStreamDestroy(stream1);

}