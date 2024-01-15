#include <iostream>
#include <random>
#include <vector>
#include <chrono>
#include <cstring>

void cuda_gmem_count_frequency(unsigned char* random_bytes, int stream_length, int* histo);
void cuda_smem_count_frequency(unsigned char* random_bytes, int stream_length, int* histo);

void generate_byte_stream(unsigned char* random_bytes, int stream_length) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);

    for (int i = 0; i < stream_length; ++i) {
        random_bytes[i] = static_cast<unsigned char>(dis(gen));
    }
}

void cpu_count_frequency(unsigned char* random_bytes, int stream_length, int* histo) {
    auto start = std::chrono::high_resolution_clock::now();

    std::memset(histo, 0, 256*sizeof(int));

    for (int i = 0; i < stream_length; ++i) {
        histo[random_bytes[i]]++;
    }

    auto end = std::chrono::high_resolution_clock::now(); 
    auto elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    printf("CPU count time elapsed: %.6f ms\n", elapsed);
}

int main() {
    int stream_length = 100*1024*1024;

    unsigned char* random_bytes = (unsigned char*)malloc(stream_length);
    int cpu_histo[256], cuda_gmem_histo[256], cuda_smem_histo[256];

    generate_byte_stream(random_bytes, stream_length);

    cpu_count_frequency(random_bytes, stream_length, cpu_histo);
    cuda_gmem_count_frequency(random_bytes, stream_length, cuda_gmem_histo);
    cuda_smem_count_frequency(random_bytes, stream_length, cuda_smem_histo);
    
    for (int i = 0; i < 256; ++i) {
        if (cpu_histo[i] != cuda_gmem_histo[i]) {
            printf("result incorrect at %d, cpu: %d, cuda gmem: %d\n", i, cpu_histo[i], cuda_gmem_histo[i]);
        }
        if (cpu_histo[i] != cuda_smem_histo[i]) {
            printf("result incorrect at %d, cpu: %d, cuda smem: %d\n", i, cpu_histo[i], cuda_smem_histo[i]);
        }
    }

    free(random_bytes);
}