#include <iostream>
#include <cuda_runtime.h>
#include <string>
#include <memory>
#include <cstring>
#include "utils/io_utils.hpp"
#include "setup/preprocessing.hpp"


void cgns_driver(const std::string& projectPath){
    std::string cgnspath=projectPath + std::string("/data/setup.cgns");
    std::string jsonpath=projectPath + std::string("/em_data/setup.json");
    IOUtils::writeJSONToCGNS(cgnspath, jsonpath);
}

int main(int argc, char* argv[]) {
    std::cout << "Number of arguments: " << argc << std::endl;
    for (int i = 0; i < argc; ++i) {
        std::cout << "Argument " << i << ": " << argv[i] << std::endl;
    }

    // Check if PROJECT_PATH is defined via CMake
#ifdef PROJECT_PATH
    const std::string projectPath = PROJECT_PATH;
    const std::string basePath = projectPath.substr(0, projectPath.find_last_of("/"));
    std::cout << "Project Path: " << projectPath << "Base Path: " << basePath << std::endl;
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


    if (command == "cgns") {
        std::cout << "Testing CGNS generation..." << std::endl;
        cgns_driver(projectPath);
    }

    std::unique_ptr<preprocessing> preproc = std::make_unique<preprocessing>();
    if (command == "DICOMPNG") {
        std::cout << "Running DICOM to PNG Conversion..." << std::endl;
        preproc->convert_dicom_to_png(basePath);
    }
    else if (command == "PNGFOLDS") {
        std::cout << "Running PNG to FOLDS Creation..." << std::endl;
        preproc->create_png_to_folds(basePath);
    }
    else if (command == "PNGFOLDS_RR") {
        std::cout << "Running PNG to FOLDS (Round Robin) Creation..." << std::endl;
        preproc->create_png_to_folds_rr(basePath);
    }
    else{
        std::cerr << "Unrecognized command to execute ..." << std::endl;
    }

    return 0;
}

