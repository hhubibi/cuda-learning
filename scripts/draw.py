import pandas as pd
import matplotlib.pyplot as plt

csv_file_path = "../data/add.csv"
data = pd.read_csv(csv_file_path)

# print(data.columns)

block_dim = data['block_dim']
cpu_exe_time = data['cpu']
cpu_omp_exe_time = data['cpu_omp']
cuda_exe_time = data['cuda']
cuda_um_exe_time = data['cuda_um']

plt.plot(block_dim, cpu_exe_time, label='cpu_exe_time', color='darkgreen', marker='x', linewidth=2, markersize=5)
plt.plot(block_dim, cpu_omp_exe_time, label='cpu_omp_exe_time', color='slategrey', marker='o', linewidth=2, markersize=5)
plt.plot(block_dim, cuda_exe_time, label='cuda_exe_time', color='firebrick', marker='^', linewidth=2, markersize=5)
plt.plot(block_dim, cuda_um_exe_time, label='cuda_um_exe_time', color='steelblue', marker='*', linewidth=2, markersize=5)

plt.xlabel('block dim')
plt.ylabel('exe time(msec)')

plt.legend()
# plt.grid(True)

plt.show()
