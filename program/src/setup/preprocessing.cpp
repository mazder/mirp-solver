#include "setup/preprocessing.hpp"

#include <cmath>
#include <numeric>
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>  // For embedding the Python interpreter
#include <string>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <opencv2/features2d.hpp>  // For ORB
#include <filesystem>
namespace fs = std::filesystem;

namespace py = pybind11;


preprocessing::preprocessing()
{
    //ctor
}

preprocessing::~preprocessing()
{
    //dtor
}

void preprocessing::dicom_to_png(const std::string& in_path,const std::string& out_path)
{
    std::string command=std::string("python3 ") + PROJECT_PATH + std::string("/src/setup/preprocess/main.py") + PROJECT_PATH;
    if (system(command.c_str()) != 0) {
        std::cerr << "Error executing Python script!" << std::endl;
        exit(0);
    }
}


