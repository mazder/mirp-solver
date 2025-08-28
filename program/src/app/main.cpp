#include <iostream>
#include <cuda_runtime.h>
#include <string>
#include <memory>
#include <cstring>


int main(int argc, char* argv[]) {
    std::cout << "Number of arguments: " << argc << std::endl;
    for (int i = 0; i < argc; ++i) {
        std::cout << "Argument " << i << ": " << argv[i] << std::endl;
    }

    // Check if PROJECT_PATH is defined via CMake
#ifdef PROJECT_PATH
    const std::string projectPath = PROJECT_PATH;
    std::cout << "Project Path: " << projectPath << std::endl;
#else
    std::cerr << "PROJECT_PATH is not defined!" << std::endl;
    return 1;
#endif

    // Ensure at least one command argument is provided
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <command>" << std::endl;
        std::cerr << "Commands: segmentation, classification, ..." << std::endl;
        return 1;
    }

    const std::string command = argv[1];
    std::cout << "Executing command <::> "<< command << std::endl;
    
    return 0;
}

