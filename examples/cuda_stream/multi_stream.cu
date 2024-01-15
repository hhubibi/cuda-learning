#include <cuda_runtime.h>
#include <stdio.h>

__global__ void multi_stream_kernel(int N, int *a, int *b, int *c) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        int idx1 = (idx + 1) % 256;
        int idx2 = (idx + 2) % 256;
        float   as = (a[idx] + a[idx1] + a[idx2]) / 3.0f;
        float   bs = (b[idx] + b[idx1] + b[idx2]) / 3.0f;
        c[idx] = (as + bs) / 2;
    }
}

void multi_stream(int stream_num) {
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

    cudaStream_t streams[stream_num];
    for (int i = 0; i < stream_num; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int *host_a, *host_b, *host_c;
    int *dev_a, *dev_b, *dev_c;

    cudaMalloc((void**)&dev_a, FULL_DATA_SIZE*sizeof(int));
    cudaMalloc((void**)&dev_b, FULL_DATA_SIZE*sizeof(int));
    cudaMalloc((void**)&dev_c, FULL_DATA_SIZE*sizeof(int));

    cudaHostAlloc((void**)&host_a, FULL_DATA_SIZE*sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void**)&host_b, FULL_DATA_SIZE*sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void**)&host_c, FULL_DATA_SIZE*sizeof(int), cudaHostAllocDefault);

    for (int i = 0; i < FULL_DATA_SIZE; ++i) {
        host_a[i] = rand();
        host_b[i] = rand();
    }

    cudaEventRecord(start, 0);

    int offset = 0;
    for (int i = 0; i < FULL_DATA_SIZE; i += N*stream_num) {
        for (int j = 0; j < stream_num; ++j) {
            offset = i + j*N;
            if (offset >= FULL_DATA_SIZE) {
                break;
            }
            cudaMemcpyAsync(dev_a+offset, host_a+offset, N*sizeof(int), cudaMemcpyHostToDevice, streams[j]);
            cudaMemcpyAsync(dev_b+offset, host_b+offset, N*sizeof(int), cudaMemcpyHostToDevice, streams[j]);
            multi_stream_kernel<<<N/256, 256, 0, streams[j]>>>(N, dev_a+offset, dev_b+offset, dev_c+offset);
            cudaMemcpyAsync(host_c+offset, dev_c+offset, N*sizeof(int), cudaMemcpyDeviceToHost, streams[j]);
        }
    }
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);
    printf( "Time taken with %d stream:  %3.1f ms\n", stream_num, elapsed);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    cudaFreeHost(host_a);
    cudaFreeHost(host_b);
    cudaFreeHost(host_c);
    
    for (int i = 0; i < stream_num; ++i) {
        cudaStreamDestroy(streams[i]);
    }

}