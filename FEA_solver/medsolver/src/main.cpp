#include "mesh_generator.h"
#include "mesh.h"
#include "fem_solver.h"
#include <iostream>   // std::cerr, std::endl
#include <string>     // std::string
#include <algorithm>  // std::transform
#include <memory>     // std::unique_ptr, std::make_unique

#include <Eigen/Dense>

#include <filesystem>
#include <iomanip>
#include <sstream>
#include <vector>

namespace fs = std::filesystem;

int driverMeshGenerator(){

    std::unique_ptr<medsolver::MeshGenerator> meshGen = std::make_unique<medsolver::MeshGenerator>();

    int number_patients=20;
    for (int i = 1; i <= number_patients; ++i) {
        // Build patient ID: EGD-0001 ... EGD-0020
        std::ostringstream idStream;
        idStream << "EGD-" << std::setw(4) << std::setfill('0') << i;
        std::string patientID = idStream.str();

        // Base directory
        std::string baseDir = "/workspace/mirp-solver/simulation_data/brain_tumar/"
                              + patientID + "/MR_" + patientID;

        // Collect all NIfTI images (from subdirs like 1_T1, 2_T1GD, 3_T2, 4_FLAIR)
        std::vector<std::string> imageFiles;
        for (const auto& dirEntry : fs::directory_iterator(baseDir)) {
            if (fs::is_directory(dirEntry)) {
                std::string subdir = dirEntry.path().filename().string();
                if (subdir.find("_MASK") != std::string::npos) continue; // skip mask dir

                std::string niiPath = dirEntry.path().string() + "/NIFTI";
                if (fs::exists(niiPath)) {
                    for (const auto& f : fs::directory_iterator(niiPath)) {
                        if (f.path().extension() == ".gz" || f.path().extension() == ".nii") {
                            imageFiles.push_back(f.path().string());
                        }
                    }
                }
            }
        }

        // Mask (always in 5_MASK)
        std::string maskPath = baseDir + "/5_MASK/NIFTI/MASK.nii.gz";

        std::cout << "Processing " << patientID << std::endl;
        std::cout << "  Found " << imageFiles.size() << " image(s)" << std::endl;
        std::cout << "  Mask : " << maskPath << std::endl;

        for (const auto& img : imageFiles) {
            std::cout << "   -> Using image: " << img << std::endl;

            if (!meshGen->LoadImageMask(img, maskPath)) {
                std::cerr << "❌ Failed to load " << img << " with mask " << maskPath << std::endl;
                continue;
            }
/*
            if (!meshGen->ConvertNiiImageMaskToMesh()) {
                std::cerr << "❌ Mesh conversion failed for " << img << std::endl;
                continue;
            }

            // Save mesh with patient + modality name
            std::string outFile = "/workspace/mirp-solver/simulation_data/brain_tumar_mesh/"
                                  + patientID + "_"
                                  + fs::path(img).parent_path().parent_path().filename().string();
            meshGen->SaveMesh(outFile, "vtu", 5);
 */

//            // Example defaults tuned for tumor masks:
//            meshGen->ConvertMaskToTetraMeshesHighQuality(
//                /*denoise=*/true,   /*gaussSigma=*/0.6,
//                /*resampleIso=*/true, /*targetIso=*/1.0,
//                /*smoothIters=*/35, /*passBand=*/0.05,
//                /*useAlpha=*/false, /*alpha=*/1e6, /*tol=*/1e-6
//            );
//
//            // Save mesh with patient + modality name
//            std::string outFile = "/workspace/mirp-solver/simulation_data/brain_tumar_mesh_fine/"
//                                  + patientID + "_"
//                                  + fs::path(img).parent_path().parent_path().filename().string();
//            meshGen->SaveMesh(outFile, "vtu", 5);


            bool success = meshGen->ConvertMaskToTetraMeshesHighQuality(
                true,       // applySmoothing
                0.1,        // smoothRelaxation (0..1, smaller = less smoothing)
                false,      // useFeatureEdgeSmoothing
                45.0,       // featureAngle (degrees, for edge preservation)
                20,         // subdivisionIterations (smoothing iterations)
                1.0,        // tetQuality (currently not used in vtkDelaunay3D, placeholder)
                false,      // applyDecimation
                0.5,        // targetReduction (for decimation, 0..1)
                1e-6        // minVolume (optional filtering of tiny tets)
            );

            if (!success) {
                std::cerr << "Failed to generate high-quality tetrahedral meshes!" << std::endl;
            } else {
                std::cout << "High-quality tetrahedral meshes successfully generated." << std::endl;
            }

            // Save mesh with patient + modality name
            std::string outFile = "/workspace/mirp-solver/simulation_data/brain_tumar_mesh_fine2/"
                                  + patientID + "_"
                                  + fs::path(img).parent_path().parent_path().filename().string();
            meshGen->SaveMesh(outFile, "vtu", 5);


        }
    }


/*
    const std::string img = "/workspace/mirp-solver/simulation_data/brain_tumar/EGD-0001/MR_EGD-0001/1_T1/NIFTI/T1.nii.gz";
    const std::string msk = "/workspace/mirp-solver/simulation_data/brain_tumar/EGD-0001/MR_EGD-0001/5_MASK/NIFTI/MASK.nii.gz";
    if (!meshGen->LoadImageMask(img,msk)) {
        std::cerr << "Failed to load data." << std::endl;
        return -1;
    }

    if (!meshGen->ConvertNiiImageMaskToMesh()) {
        std::cerr << "Failed to convert to mesh." << std::endl;
        return -1;
    }

    std::string filename = "./mesh_";
    if (!meshGen->SaveMesh(filename, "vtu")) {
        std::cerr << "Failed to save mesh: " << filename << std::endl;
    }
*/

    return 0;
}

int main(int argc, char** argv) {

    if (argc < 2) {
        std::cerr << "Usage: medsolver <command>"<< argv[0] << std::endl;
        return 1;
    }

    driverMeshGenerator();

    return 0;
}
