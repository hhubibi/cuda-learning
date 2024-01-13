#include <iostream>
#include <cmath>
#include <fstream>
#include <filesystem>

void print_device_msg();
float cpu_add(bool use_omp);
float cuda_add(int block_dim);
float cuda_add_um(int block_dim);

int main(int argc, char *argv[]) {
    print_device_msg();

    std::filesystem::path file_path(__FILE__);
    std::string dir_path = file_path.parent_path().parent_path().parent_path().string();
    std::string csv_file_path = dir_path + "/data/add.csv";
    std::ofstream csv_file(csv_file_path);
    if (!csv_file.is_open()) {
        throw std::runtime_error("open csv file failed");
    }

    csv_file << "block_dim,cpu,cpu_omp,cuda,cuda_um\n";
    for (int block_dim = 32; block_dim <= (1<<10); block_dim += 32) {
        csv_file << block_dim << ',';
        csv_file << cpu_add(false) << ',';
        csv_file << cpu_add(true) << ',';
        csv_file << cuda_add(block_dim) << ',';
        csv_file << cuda_add_um(block_dim) << '\n';
    }
    csv_file.close();

    return 0;
}