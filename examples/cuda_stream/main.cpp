#include <iostream>

float cuda_malloc_test(int test_size, bool up);
float cuda_host_alloc_test(int test_size, bool up);

void single_stream();
void double_stream();
void double_stream_correct();

void malloc_test() {
    int test_size = 64*1024*1024;
    float elapsed;
    float MB = (float)100*test_size*sizeof(int)/1024/1024;

    // try it with cudaMalloc
    elapsed = cuda_malloc_test( test_size, true );
    printf( "Time using cudaMalloc:  %3.1f ms\n",
            elapsed );
    printf( "\tMB/s during copy up:  %3.1f\n",
            MB/(elapsed/1000) );

    elapsed = cuda_malloc_test( test_size, false );
    printf( "Time using cudaMalloc:  %3.1f ms\n",
            elapsed );
    printf( "\tMB/s during copy down:  %3.1f\n",
            MB/(elapsed/1000) );

    // now try it with cudaHostAlloc
    elapsed = cuda_host_alloc_test( test_size, true );
    printf( "Time using cudaHostAlloc:  %3.1f ms\n",
            elapsed );
    printf( "\tMB/s during copy up:  %3.1f\n",
            MB/(elapsed/1000) );

    elapsed = cuda_host_alloc_test( test_size, false );
    printf( "Time using cudaHostAlloc:  %3.1f ms\n",
            elapsed );
    printf( "\tMB/s during copy down:  %3.1f\n",
            MB/(elapsed/1000) );
}

int main() {
    // malloc_test();

    single_stream();

    double_stream();

    double_stream_correct();
}