#include <iostream>
#include <cuda_runtime.h>
#include <string>
#include "utils/io_utils.hpp"

using namespace std;

int main(int argc, char* argv[]) {

    // Use the PROJECT_PATH passed from CMake
    #ifdef PROJECT_PATH
        std::cout << "Project Path: " << PROJECT_PATH << std::endl;
    #else
        std::cout << "PROJECT_PATH is not defined!" << std::endl;
    #endif

    std::string cgnspath=PROJECT_PATH + std::string("/data/setup.cgns");
    std::string jsonpath=PROJECT_PATH + std::string("/data/setup.json");

    IOUtils::writeJSONToCGNS(cgnspath, jsonpath);
}

