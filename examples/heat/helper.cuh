#ifndef __HELPER_CUH__
#define __HELPER_CUH__

#include <cuda_runtime.h>

struct DataBlock {
    uchar4* bitmap_ptr;
    float *in_ptr;
    float *out_ptr;
    float *const_ptr;

    cudaEvent_t start, end;
    float total_time;
    float frames;

    cudaTextureObject_t tex_const, tex_in, tex_out;
};

const int DIM = 1024;
const float SPEED = 0.25f;
const float MAX_TEMP = 1.0f;
const float MIN_TEMP = 0.0001f;

__device__ unsigned char value( float n1, float n2, int hue );

__global__ void float_to_color( uchar4 *optr,
                              const float *outSrc );

#endif