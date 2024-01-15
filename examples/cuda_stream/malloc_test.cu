#include <cuda_runtime.h>
#include <stdio.h>

float cuda_malloc_test(int sz, bool up) {
    int *a, *dev_a;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    a = (int*)malloc(sz * sizeof(int));
    cudaMalloc((void**)&dev_a, sz*sizeof(int));

    cudaEventRecord(start, 0);
    for (int i = 0; i < 100; ++i) {
        if (up) {
            cudaMemcpy(dev_a, a, sz * sizeof(int), cudaMemcpyHostToDevice);
        } else {
            cudaMemcpy(a, dev_a, sz * sizeof(int), cudaMemcpyDeviceToHost);
        }
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(dev_a);
    free(a);

    return elapsed;
}

float cuda_host_alloc_test(int sz, bool up) {
    int *a, *dev_a;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaHostAlloc((void**)&a, sz*sizeof(int), cudaHostAllocDefault);
    cudaMalloc((void**)&dev_a, sz*sizeof(int));

    cudaEventRecord(start, 0);
    for (int i = 0; i < 100; ++i) {
        if (up) {
            cudaMemcpy(dev_a, a, sz * sizeof(int), cudaMemcpyHostToDevice);
        } else {
            cudaMemcpy(a, dev_a, sz * sizeof(int), cudaMemcpyDeviceToHost);
        }
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(dev_a);
    cudaFreeHost(a);

    return elapsed;
}
