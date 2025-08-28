#include <iostream>
#include <vector>
#include <string>
#include <yaml-cpp/yaml.h>
#include <SimpleITK.h>

#define FILENAME "example.cgns"

// Function to load the image data (NIfTI or other formats)
SimpleITK::Image loadImage(const std::string& image_path) {
    try {
        SimpleITK::Image image = SimpleITK::ReadImage(image_path);
        return image;
    } catch (const std::exception& e) {
        std::cerr << "Error loading image: " << e.what() << std::endl;
        throw;
    }
}

// Function to apply image augmentation (for simplicity, we just flip the image)
SimpleITK::Image augmentImage(const SimpleITK::Image& image) {
    // Example augmentation: flip the image along the X axis
    return SimpleITK::Flip(image, {true, false, false});
}

// Function to process and display results
void processAndSaveResults(const SimpleITK::Image& image, const std::string& output_dir) {
    try {
        // Save the augmented image to the output directory
        std::string output_path = output_dir + "/augmented_image.nii";
        SimpleITK::WriteImage(image, output_path);
        std::cout << "Saved augmented image to: " << output_path << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error saving results: " << e.what() << std::endl;
    }
}

int main() {
    try {
        // Load YAML configuration
        YAML::Node config = YAML::LoadFile("../config.yaml");

        // Retrieve input data paths
        std::string image_path = config["input_data"]["image_path"].as<std::string>();
        std::string tumor_segmentation_mask_path = config["input_data"]["tumor_segmentation_mask_path"].as<std::string>();
        std::string output_dir = config["output"]["visualization"]["output_directory"].as<std::string>();

        // Load the image
        std::cout << "Loading image from: " << image_path << std::endl;
        SimpleITK::Image image = loadImage(image_path);

        // Apply augmentation (for example, flipping the image)
        std::cout << "Applying augmentations..." << std::endl;
        SimpleITK::Image augmented_image = augmentImage(image);

        // Process and save results
        std::cout << "Processing and saving results..." << std::endl;
        processAndSaveResults(augmented_image, output_dir);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
