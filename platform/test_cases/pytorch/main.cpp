#include <torch/torch.h>
#include <iostream>
#include <cudnn.h>

int main() {

    cudnnHandle_t handle;
    cudnnCreate(&handle);
    std::cout << "cuDNN version: " << CUDNN_VERSION << std::endl;
    cudnnDestroy(handle);

    // Ensure that CUDA is available on your system
    if (!torch::cuda::is_available()) {
        std::cerr << "CUDA is not available! Please check your CUDA setup." << std::endl;
        return -1;
    }

    // Move tensors to GPU
    torch::Tensor a = torch::rand({2, 3}, torch::device(torch::kCUDA));  // 2x3 matrix on GPU
    torch::Tensor b = torch::rand({3, 2}, torch::device(torch::kCUDA));  // 3x2 matrix on GPU

    // Perform matrix multiplication on the GPU (2x3) * (3x2) = (2x2)
    torch::Tensor result = torch::matmul(a, b);

    // Print the result
    std::cout << "Tensor A: \n" << a << std::endl;
    std::cout << "Tensor B: \n" << b << std::endl;
    std::cout << "Result of Matrix Multiplication (on GPU): \n" << result << std::endl;

    return 0;
}

