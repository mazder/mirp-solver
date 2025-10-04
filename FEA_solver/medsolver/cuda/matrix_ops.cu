#include <cuda_runtime.h>
#include <iostream>

__global__ void matmul_kernel() {
    printf("Matrix multiplication kernel!\n");
}

extern "C" void launch_matrix_ops() {
    matmul_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
}
