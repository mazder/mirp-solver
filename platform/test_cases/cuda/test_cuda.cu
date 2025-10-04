#include <iostream>
#include <cmath>

#define N 10  // Size of the vectors

// CUDA kernel to add two vectors
__global__ void vectorAdd(int *A, int *B, int *C, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    int A[N], B[N], C[N];  // Host vectors
    int *d_A, *d_B, *d_C;  // Device vectors

    // Initialize input vectors on the host
    for (int i = 0; i < N; i++) {
        A[i] = i;
        B[i] = i * 2;
    }

    // Allocate memory on the device
    cudaMalloc((void**)&d_A, N * sizeof(int));
    cudaMalloc((void**)&d_B, N * sizeof(int));
    cudaMalloc((void**)&d_C, N * sizeof(int));

    // Copy input vectors from host to device
    cudaMemcpy(d_A, A, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel with one block of 10 threads (one for each element)
    vectorAdd<<<1, N>>>(d_A, d_B, d_C, N);

    // Copy result back to host
    cudaMemcpy(C, d_C, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the result
    std::cout << "Result of vector addition:" << std::endl;
    for (int i = 0; i < N; i++) {
        std::cout << A[i] << " + " << B[i] << " = " << C[i] << std::endl;
    }

    // Free the memory on the device
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
