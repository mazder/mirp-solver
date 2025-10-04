#include <iostream>
#include <cudnn.h>

int main() {
    cudnnHandle_t handle;
    cudnnCreate(&handle);
    std::cout << "cuDNN version: " << CUDNN_VERSION << std::endl;
    cudnnDestroy(handle);
    return 0;
}
