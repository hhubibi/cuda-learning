#include <cuda_runtime.h>
#include <stdio.h>
#include <cuComplex.h>

__global__ void julia(uchar3* ptr, int width, int height, float scale, int max_iter, cuComplex c, int ticks) {
    int x = blockIdx.x;
    int y = blockIdx.y;
    int offset = x + y * gridDim.x;
    
    if (x < width && y < height) {
        float jx = scale * (float)(width/2 - x) / (width/2);
        float jy = scale * (float)(height/2 - y) / (height/2);
        cuComplex a = make_cuComplex(jx, jy);

        int i = 0;
        for (; i < max_iter; ++i) {
            a = cuCaddf(cuCmulf(a, a), c);
            if (cuCabsf(a) > 10.0f) {
                break;
            }
        }

        // int flag = (i == max_iter) ? 1 : 0;
        float d = cuCabsf(a);
        unsigned char grey = (unsigned char)(128.0f + 127.0f * cos(d/10.0f-ticks/7.0f)/(d/10.0f+1.0f));

        ptr[offset] = make_uchar3(grey, grey, 0);
    }   
}



void cuda_draw_julia(uchar3* ptr, int width, int height, float scale, int max_iter, float c_x, float c_y, int ticks) {
    dim3 block_dim(1);
    dim3 grid_dim(width, height);

    cuComplex c = make_cuComplex(c_x, c_y);
    julia<<<grid_dim, block_dim>>>(ptr, width, height, scale, max_iter, c, ticks);
}