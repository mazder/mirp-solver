#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>

int main() {
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;

    // Check if CUDA is available
    if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
        std::cout << "CUDA is available" << std::endl;
    } else {
        std::cout << "CUDA is not available" << std::endl;
    }

    // Load an image using OpenCV
    cv::Mat img = cv::imread("test.jpeg");  // Replace with your image path
    if (img.empty()) {
        std::cerr << "Could not open or find the image!" << std::endl;
        return -1;
    }

    // Save the image to a file instead of displaying it
    bool result = cv::imwrite("output_image.jpg", img);  // Save as output_image.jpg
    if (!result) {
        std::cerr << "Could not save the image!" << std::endl;
        return -1;
    }

    std::cout << "Image has been saved as output_image.jpg" << std::endl;

    return 0;
}
