#include "mesh_generator.h"

#include <iostream>
#include <filesystem>

#include <vtkNIFTIImageReader.h>
#include <vtkDICOMImageReader.h>

#include <vtkXMLUnstructuredGridWriter.h>
#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>
#include <vtkboneImageToMesh.h>
#include <vtkZLibDataCompressor.h>

#include <algorithm>                // For std::clamp
#include <sstream>                  // For std::ostringstream
#include <vtkImageThreshold.h>


// Headers you’ll likely need:
#include <vtkSmartPointer.h>
#include <vtkImageData.h>
#include <vtkUnstructuredGrid.h>
#include <vtkImageCast.h>
#include <vtkImageGaussianSmooth.h>
#include <vtkImageMedian3D.h>
#include <vtkImageResample.h>
#include <vtkDiscreteMarchingCubes.h>
#include <vtkWindowedSincPolyDataFilter.h>
#include <vtkSmoothPolyDataFilter.h>
#include <vtkTriangleFilter.h>
#include <vtkCleanPolyData.h>
#include <vtkPolyDataNormals.h>
#include <vtkDelaunay3D.h>
#include <vtkGeometryFilter.h>           // if you need to visualize UG as surface
#include <vtkPointData.h>
#include <vtkDataArray.h>


// Include other VTK headers you use for mesh generation and saving
namespace fs = std::filesystem;

namespace medsolver {

MeshGenerator::MeshGenerator() {}

MeshGenerator::~MeshGenerator() {}

// ------------------------------------------------------------
// LoadImageMask
//  - Loads an image and a mask file (e.g. CT + segmentation).
//  - Verifies that both paths are valid files.
//  - Delegates to LoadStoreImageMask() for actual reading.
// ------------------------------------------------------------
bool MeshGenerator::LoadImageMask(const std::string& imPath, const std::string& mskPath) {
    try {
        // Check both input paths are valid files
        if (fs::is_regular_file(imPath) && fs::is_regular_file(mskPath)) {
            std::cout << "Loading single files: " << imPath << " and " << mskPath << std::endl;

            // Load image (imgf=1, mskf=0) and mask (imgf=0, mskf=1)
            return LoadStoreImageMask(imPath, true, false) &&
                   LoadStoreImageMask(mskPath, false, true);
        }
        else {
            std::cerr << "Invalid file(s). imPath: " << imPath
                      << "  mskPath: " << mskPath << std::endl;
            return false;
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Exception during LoadImageMask: " << e.what() << std::endl;
        return false;
    }
}


// ------------------------------------------------------------
// LoadStoreImageMask
//  - Loads a single file (NIfTI supported).
//  - Stores it into `images_` if it's an image,
//    or into `masks_` if it's a mask.
// ------------------------------------------------------------
bool MeshGenerator::LoadStoreImageMask(const std::string& filename, bool imgf, bool mskf) {
    // Detect file type by extension (currently supports only NIfTI)
    std::string ext = fs::path(filename).extension().string();
    if (ext == ".nii" || ext == ".gz") {
        // Create and configure NIfTI reader
        vtkSmartPointer<vtkNIFTIImageReader> reader = vtkSmartPointer<vtkNIFTIImageReader>::New();
        reader->SetFileName(filename.c_str());
        reader->Update();

        vtkSmartPointer<vtkImageData> img = reader->GetOutput();

        if (!img) {
            std::cerr << "Failed to load NIfTI file: " << filename << std::endl;
            return false;
        }

        // Store into appropriate container
        if (imgf && !mskf) {
            images_.push_back(img);
        }
        else if (!imgf && mskf) {
            masks_.push_back(img);
        }
        else {
            std::cerr << "Ambiguous flags for file: " << filename << std::endl;
            return false;
        }

        return true;
    }
    else {
        std::cerr << "Unsupported file extension: " << ext
                  << " for file: " << filename << std::endl;
        return false;
    }
}

// ------------------------------------------------------------
// ConvertNiiImageMaskToMesh
//  - Converts loaded NIfTI image/mask pairs into finite element meshes.
//  - Iterates over available (image, mask) pairs.
//  - For each mask, extracts labeled regions > 0.
//  - Uses vtkboneImageToMesh to create meshes for each label.
//  - Stores resulting meshes in `meshes_`.
// ------------------------------------------------------------
bool MeshGenerator::ConvertNiiImageMaskToMesh() {
    // Debug: report how many images/masks are currently loaded
    std::cerr << "Images loaded: " << images_.size()
              << ", Masks loaded: " << masks_.size() << std::endl;

    // Safety check: require at least one image and one mask
    if (images_.empty() || masks_.empty()) {
        std::cerr << "Either images or masks (or both) are not loaded for meshing." << std::endl;
        return false;
    }

    // Reset meshes from any previous run
    meshes_.clear();

    // We can only process up to the smaller of (numImages, numMasks)
    size_t nPairs = std::min(images_.size(), masks_.size());

    // Loop over image–mask pairs
    for (size_t i = 0; i < nPairs; ++i) {
        vtkImageData* seg = masks_[i];
        if (!seg) {
            std::cerr << "Null mask at index " << i << ", skipping." << std::endl;
            continue;
        }

        // Determine label range in mask (e.g., background=0, regions >0)
        double range[2];
        seg->GetScalarRange(range);
        int minLabel = static_cast<int>(range[0]);
        int maxLabel = static_cast<int>(range[1]);

        // Iterate over labels found in the mask
        for (int label = minLabel; label <= maxLabel; ++label) {
            if (label <= 0) continue;  // Skip background or invalid labels

            // --------------------------------------------------------
            // Step 1: Threshold mask to isolate current label
            // --------------------------------------------------------
            auto thresh = vtkSmartPointer<vtkImageThreshold>::New();
            thresh->SetInputData(seg);
            thresh->ThresholdBetween(label, label);   // keep only voxels == label
            thresh->SetInValue(1);                    // foreground
            thresh->SetOutValue(0);                   // background
            thresh->SetOutputScalarTypeToUnsignedChar();
            thresh->Update();

            // --------------------------------------------------------
            // Step 2: Convert binary mask into a mesh
            // --------------------------------------------------------
            auto imageToMesh = vtkSmartPointer<vtkboneImageToMesh>::New();
            imageToMesh->SetInputConnection(thresh->GetOutputPort());
            imageToMesh->Update();

            // Get raw mesh output
            vtkUnstructuredGrid* rawMesh = imageToMesh->GetOutput();
            if (!rawMesh || rawMesh->GetNumberOfCells() == 0) {
                std::cerr << "⚠️  No mesh generated for label " << label
                          << " in mask index " << i << std::endl;
                continue;
            }

            // --------------------------------------------------------
            // Step 3: Store mesh safely
            // --------------------------------------------------------
            auto mesh = vtkSmartPointer<vtkUnstructuredGrid>::New();
            mesh->ShallowCopy(rawMesh);
            meshes_.push_back(mesh);

            std::cout << "✅ Mesh created for label " << label
                      << " in mask index " << i << std::endl;
        }
    }

    // Report total number of meshes created
    std::cout << "Total meshes created: " << meshes_.size() << std::endl;

    // clear
    images_.clear();
    masks_.clear();

    return true;
}

// Requires VTK modules:
// CommonCore, CommonDataModel, FiltersCore, FiltersGeneral,
// FiltersGeometry, FiltersModeling, FiltersSources, ImagingCore
// (and optionally vtkvmtk or TetGen if you later swap tetrahedralizer)

bool MeshGenerator::ConvertMaskToTetraMeshesHighQuality(
    // --- Preprocessing (optional) ---
    bool     denoise      /*= true*/,
    double   gaussSigma   /*= 0.5*/,    // voxel units
    bool     resampleIso  /*= true*/,
    double   targetIso    /*= 1.0*/,    // mm/voxel target (set to your data spacing)
    // --- Surface smoothing ---
    int      smoothIters  /*= 30*/,
    double   passBand     /*= 0.04*/,   // (0..2), lower = more smoothing
    // --- Tetrahedralization (vtkDelaunay3D) ---
    bool     useAlpha     /*= false*/,
    double   alpha        /*= 1e6*/,    // only used if useAlpha==true
    double   tol          /*= 1e-6*/
)
{
    std::cerr << "[HQ] Masks: " << masks_.size() << std::endl;
    if (masks_.empty()) {
        std::cerr << "[HQ] No masks loaded." << std::endl;
        return false;
    }

    meshes_.clear();

    // Process each mask volume independently
    for (size_t i = 0; i < masks_.size(); ++i) {
        vtkImageData* maskImg = masks_[i];
        if (!maskImg) {
            std::cerr << "[HQ] Null mask at index " << i << ", skipping." << std::endl;
            continue;
        }

        // ---------------------------------------------------------------------
        // 0) OPTIONAL: pre-denoise & resample to isotropic spacing
        // ---------------------------------------------------------------------
        vtkSmartPointer<vtkImageData> workImg = maskImg;

        if (denoise) {
            auto gauss = vtkSmartPointer<vtkImageGaussianSmooth>::New();
            gauss->SetInputData(workImg);
            gauss->SetStandardDeviations(gaussSigma, gaussSigma, gaussSigma);
            gauss->SetRadiusFactors(1.0, 1.0, 1.0);
            gauss->Update();
            workImg = gauss->GetOutput();
        }

        if (resampleIso) {
            double spacing[3];
            workImg->GetSpacing(spacing);
            auto rs = vtkSmartPointer<vtkImageResample>::New();
            rs->SetInputData(workImg);
            rs->SetInterpolationModeToNearestNeighbor(); // keep labels discrete
            rs->SetOutputSpacing(targetIso, targetIso, targetIso);
            rs->Update();
            workImg = rs->GetOutput();
        }

        // Ensure unsigned char for marching cubes consistency
        auto caster = vtkSmartPointer<vtkImageCast>::New();
        caster->SetInputData(workImg);
        caster->SetOutputScalarTypeToUnsignedChar();
        caster->Update();
        workImg = caster->GetOutput();

        // ---------------------------------------------------------------------
        // 1) Determine label range
        // ---------------------------------------------------------------------
        double range[2];
        workImg->GetScalarRange(range);
        const int minLabel = static_cast<int>(range[0]);
        const int maxLabel = static_cast<int>(range[1]);

        // ---------------------------------------------------------------------
        // 2) Loop over labels (skip background 0)
        // ---------------------------------------------------------------------
        for (int label = std::max(1, minLabel); label <= maxLabel; ++label) {

            // ---- 2a) Extract surface with Discrete Marching Cubes ----
            auto dmc = vtkSmartPointer<vtkDiscreteMarchingCubes>::New();
            dmc->SetInputData(workImg);
            dmc->SetValue(0, label);                 // extract this label as a surface
            dmc->ComputeNormalsOff();
            dmc->ComputeGradientsOff();
            dmc->Update();

            if (!dmc->GetOutput() || dmc->GetOutput()->GetNumberOfCells() == 0) {
                std::cerr << "[HQ] Label " << label << " yielded empty surface. Skipping." << std::endl;
                continue;
            }

            // ---- 2b) Clean triangles & ensure manifold-ish surface ----
            // Triangulate (some filters expect triangles) + remove degenerate/duplicates
            auto tri = vtkSmartPointer<vtkTriangleFilter>::New();
            tri->SetInputConnection(dmc->GetOutputPort());
            tri->Update();

            auto clean = vtkSmartPointer<vtkCleanPolyData>::New();
            clean->SetInputConnection(tri->GetOutputPort());
            clean->SetTolerance(1e-6);
            clean->ConvertStripsToPolysOn();
            clean->ConvertPolysToLinesOff();
            clean->PointMergingOn();
            clean->Update();

            // ---- 2c) Smooth surface (windowed sinc preserves volume better) ----
            auto smooth = vtkSmartPointer<vtkWindowedSincPolyDataFilter>::New();
            smooth->SetInputConnection(clean->GetOutputPort());
            smooth->SetNumberOfIterations(smoothIters); // 20–60 is typical
            smooth->SetPassBand(passBand);              // 0.01–0.1 typical
            smooth->BoundarySmoothingOff();             // preserve edges
            smooth->FeatureEdgeSmoothingOff();
            smooth->NonManifoldSmoothingOn();
            smooth->NormalizeCoordinatesOn();
            smooth->Update();

            // Optional: recompute normals (useful for visualization)
            auto norms = vtkSmartPointer<vtkPolyDataNormals>::New();
            norms->SetInputConnection(smooth->GetOutputPort());
            norms->SetFeatureAngle(60.0);
            norms->ConsistencyOn();
            norms->SplittingOff();
            norms->AutoOrientNormalsOn();
            norms->Update();

            // -----------------------------------------------------------------
            // 3) Tetrahedralize the smoothed surface
            // -----------------------------------------------------------------
            // NOTE: vtkDelaunay3D is an *unconstrained* tetrahedralizer; it
            // fills the convex hull of the input point set. For highly concave
            // shapes (tumors often are), prefer TetGen/CGAL/VMTK for a
            // constrained tetrahedral mesh. This is a good in-VTK baseline.
            auto del = vtkSmartPointer<vtkDelaunay3D>::New();
            del->SetInputConnection(norms->GetOutputPort());
            del->SetTolerance(tol);
            del->BoundingTriangulationOff();
            if (useAlpha) {
                del->SetAlpha(alpha);   // filter tets outside local radius
                del->AlphaTrisOff();
                del->AlphaTetsOn();
            }
            del->Update();

            vtkUnstructuredGrid* ug = del->GetOutput();
            if (!ug || ug->GetNumberOfCells() == 0) {
                std::cerr << "[HQ] Delaunay produced empty UG for label " << label << ". Skipping." << std::endl;
                continue;
            }

            // Store a copy in our meshes_ vector
            auto mesh = vtkSmartPointer<vtkUnstructuredGrid>::New();
            mesh->ShallowCopy(ug);
            meshes_.push_back(mesh);

            std::cout << "[HQ] ✅ Tetra mesh created for label " << label
                      << " (cells=" << mesh->GetNumberOfCells() << ")\n";
        }
    }

    std::cout << "[HQ] Total tetra meshes created: " << meshes_.size() << std::endl;

    // clear
    images_.clear();
    masks_.clear();

    return !meshes_.empty();
}

bool MeshGenerator::ConvertMaskToTetraMeshesHighQuality2(
    bool applySmoothing,
    double smoothRelaxation,
    bool useFeatureEdgeSmoothing,
    double featureAngle,
    int subdivisionIterations,
    double tetQuality,
    bool applyDecimation,
    double targetReduction,
    double minVolume
) {
    meshes_.clear();

    for (size_t i = 0; i < masks_.size(); ++i) {
        vtkImageData* mask = masks_[i];
        if (!mask) continue;

        double range[2];
        mask->GetScalarRange(range);
        int minLabel = static_cast<int>(range[0]);
        int maxLabel = static_cast<int>(range[1]);

        for (int label = minLabel; label <= maxLabel; ++label) {
            if (label <= 0) continue;

            // -------------------------
            // Step 1: Extract surface
            // -------------------------
            auto thresh = vtkSmartPointer<vtkImageThreshold>::New();
            thresh->SetInputData(mask);
            thresh->ThresholdBetween(label, label);
            thresh->SetInValue(1);
            thresh->SetOutValue(0);
            thresh->SetOutputScalarTypeToUnsignedChar();
            thresh->Update();

            auto surfaceExtractor = vtkSmartPointer<vtkDiscreteMarchingCubes>::New();
            surfaceExtractor->SetInputConnection(thresh->GetOutputPort());
            surfaceExtractor->SetValue(0, 1);  // iso-value for binary mask
            surfaceExtractor->Update();

            vtkSmartPointer<vtkPolyData> polySurface = surfaceExtractor->GetOutput();

            // -------------------------
            // Step 2: Optional smoothing
            // -------------------------
            if (applySmoothing) {
                auto smoother = vtkSmartPointer<vtkSmoothPolyDataFilter>::New();
                smoother->SetInputData(polySurface);
                smoother->SetNumberOfIterations(subdivisionIterations);
                smoother->SetRelaxationFactor(smoothRelaxation);
                smoother->FeatureEdgeSmoothingOff();
                if (useFeatureEdgeSmoothing)
                    smoother->FeatureEdgeSmoothingOn();
                smoother->Update();
                polySurface = smoother->GetOutput();
            }

            // -------------------------
            // Step 3: Tetrahedralization
            // -------------------------
            auto delaunay = vtkSmartPointer<vtkDelaunay3D>::New();
            delaunay->SetInputData(polySurface);
            delaunay->SetTolerance(0.001);
            delaunay->SetAlpha(0);  // keeps only inside volume
            delaunay->Update();

            vtkUnstructuredGrid* tetMesh = delaunay->GetOutput();
            if (!tetMesh || tetMesh->GetNumberOfCells() == 0) {
                std::cerr << "⚠️ No tetrahedral mesh generated for label " << label
                          << " in mask " << i << std::endl;
                continue;
            }

            auto mesh = vtkSmartPointer<vtkUnstructuredGrid>::New();
            mesh->ShallowCopy(tetMesh);
            meshes_.push_back(mesh);

            std::cout << "✅ High-quality tetrahedral mesh created for label " << label
                      << " in mask " << i << std::endl;
        }
    }

    std::cout << "Total high-quality tetrahedral meshes created: " << meshes_.size() << std::endl;

    // clear
    images_.clear();
    masks_.clear();

    return true;
}

//bool MeshGenerator::ConvertMaskToTetraMeshesHighQuality(
//    bool applyGaussian = true, double sigma = 1.0,
//    bool applySmoothing = true, double relaxFactor = 0.1,
//    int iterations = 30, double alpha = 0.0,
//    bool preserveTopology = true, double tolerance = 0.0,
//    double offset = 0.0)
//{
//    std::cerr << "High-quality tetrahedral mesh generation started." << std::endl;
//
//    if (masks_.empty()) {
//        std::cerr << "No masks loaded." << std::endl;
//        return false;
//    }
//
//    meshes_.clear();
//
//    for (size_t i = 0; i < masks_.size(); ++i) {
//        vtkImageData* mask = masks_[i];
//        if (!mask) {
//            std::cerr << "Null mask at index " << i << ", skipping." << std::endl;
//            continue;
//        }
//
//        double range[2];
//        mask->GetScalarRange(range);
//        int minLabel = static_cast<int>(range[0]);
//        int maxLabel = static_cast<int>(range[1]);
//
//        for (int label = minLabel; label <= maxLabel; ++label) {
//            if (label <= 0) continue; // skip background
//
//            // -----------------------------------------------
//            // Step 1: Threshold mask for current label
//            // -----------------------------------------------
//            auto thresh = vtkSmartPointer<vtkImageThreshold>::New();
//            thresh->SetInputData(mask);
//            thresh->ThresholdBetween(label, label);
//            thresh->SetInValue(1);
//            thresh->SetOutValue(0);
//            thresh->SetOutputScalarTypeToUnsignedChar();
//            thresh->Update();
//
//            // -----------------------------------------------
//            // Step 2: Optional Gaussian smoothing
//            // -----------------------------------------------
//            vtkSmartPointer<vtkImageData> smoothedMask = thresh->GetOutput();
//            if (applyGaussian) {
//                auto gaussian = vtkSmartPointer<vtkImageGaussianSmooth>::New();
//                gaussian->SetInputData(smoothedMask);
//                gaussian->SetStandardDeviations(sigma, sigma, sigma);
//                gaussian->Update();
//                smoothedMask = gaussian->GetOutput();
//            }
//
//            // -----------------------------------------------
//            // Step 3: Surface extraction
//            // -----------------------------------------------
//            auto dmc = vtkSmartPointer<vtkDiscreteMarchingCubes>::New();
//            dmc->SetInputData(smoothedMask);
//            dmc->GenerateValues(1, 1, 1); // extract label
//            dmc->Update();
//
//            vtkSmartPointer<vtkPolyData> surface = dmc->GetOutput();
//
//            // -----------------------------------------------
//            // Step 4: Optional surface smoothing
//            // -----------------------------------------------
//            if (applySmoothing) {
//                auto smoother = vtkSmartPointer<vtkSmoothPolyDataFilter>::New();
//                smoother->SetInputData(surface);
//                smoother->SetRelaxationFactor(relaxFactor);
//                smoother->SetNumberOfIterations(iterations);
//                smoother->FeatureEdgeSmoothingOff();
//                smoother->BoundarySmoothingOn();
//                smoother->Update();
//                surface = smoother->GetOutput();
//            }
//
//            // -----------------------------------------------
//            // Step 5: Tetrahedralization (volumetric mesh)
//            // -----------------------------------------------
//            auto delaunay = vtkSmartPointer<vtkDelaunay3D>::New();
//            delaunay->SetInputData(surface);
//            delaunay->SetAlpha(alpha);               // optional alpha shape
//            delaunay->SetTolerance(tolerance);
//            delaunay->Update();
//
//            vtkUnstructuredGrid* tetMesh = delaunay->GetOutput();
//            if (!tetMesh || tetMesh->GetNumberOfCells() == 0) {
//                std::cerr << "⚠️  Failed to generate tetrahedral mesh for label "
//                          << label << " in mask " << i << std::endl;
//                continue;
//            }
//
//            auto mesh = vtkSmartPointer<vtkUnstructuredGrid>::New();
//            mesh->ShallowCopy(tetMesh);
//            meshes_.push_back(mesh);
//
//            std::cout << "✅ High-quality tetrahedral mesh created for label "
//                      << label << " in mask " << i << std::endl;
//        }
//    }
//
//    std::cout << "Total tetrahedral meshes created: " << meshes_.size() << std::endl;
//    // clear
//    images_.clear();
//    masks_.clear();
//    return true;
//}

// ------------------------------------------------------------------
// SaveMesh
//   Saves all meshes in `meshes_` to disk.
//   - Currently supports only VTU format (XML Unstructured Grid).
//   - Writes multiple meshes by appending an index to the filename.
//   - Applies binary encoding and zlib compression (if available).
// ------------------------------------------------------------------
bool MeshGenerator::SaveMesh(const std::string& filename,
                             const std::string& format,
                             int compressionLevel) {

    std::cout << "Number of meshes: " << meshes_.size() << std::endl;

    // Safety: ensure we actually have meshes to save
    if (meshes_.empty()) {
        std::cerr << "❌ Error: No meshes to save." << std::endl;
        return false;
    }

    // Only support VTU for now
    if (format != "vtu") {
        std::cerr << "❌ Unsupported format: " << format
                  << " (only 'vtu' supported)" << std::endl;
        return false;
    }

    int count = 0;
    for (const auto& mesh : meshes_) {
        if (!mesh) {
            std::cerr << "⚠️  Warning: Skipping null mesh at index "
                      << count << std::endl;
            ++count;
            continue;
        }

        // Create a writer for XML Unstructured Grid
        auto writer = vtkSmartPointer<vtkXMLUnstructuredGridWriter>::New();

        // Construct output filename: e.g., base_0.vtu, base_1.vtu
        std::ostringstream oss;
        oss << filename << "_" << count << ".vtu";
        writer->SetFileName(oss.str().c_str());

        // Set mesh input
        writer->SetInputData(mesh);

        // Save as binary (smaller + faster to load than ASCII)
        writer->SetDataModeToBinary();

        // Clamp compression level (valid: 0–9)
        compressionLevel = std::clamp(compressionLevel, 0, 9);
        writer->SetCompressionLevel(compressionLevel);

        // Configure compressor depending on VTK version
    #if VTK_MAJOR_VERSION >= 9
        auto compressor = vtkSmartPointer<vtkZLibDataCompressor>::New();
        writer->SetCompressor(compressor);
    #else
        writer->SetCompressorTypeToZLib();
    #endif

        // Attempt to write mesh
        if (!writer->Write()) {
            std::cerr << "❌ Failed to write mesh to file: "
                      << oss.str() << std::endl;
            return false;
        }

        std::cout << "✅ Mesh saved: " << oss.str() << std::endl;
        ++count;
    }

    return true;
}

//bool MeshGenerator::SaveHighQualityMesh(const std::string& baseDir,
//                                        const std::string& patientID,
//                                        const std::string& modality,
//                                        int compressionLevel)
//{
//    if (meshes_.empty()) {
//        std::cerr << "Error: No high-quality meshes to save." << std::endl;
//        return false;
//    }
//
//    // Clamp compression level to valid range
//    compressionLevel = std::clamp(compressionLevel, 0, 9);
//
//    int count = 0;
//    for (const auto& mesh : meshes_) {
//        if (!mesh) {
//            std::cerr << "Warning: Skipping null mesh at index " << count << std::endl;
//            ++count;
//            continue;
//        }
//
//        // Construct output filename: patientID_modality_index.vtu
//        std::ostringstream oss;
//        oss << baseDir << "/"
//            << patientID << "_"
//            << modality << "_"
//            << count << ".vtu";
//
//        auto writer = vtkSmartPointer<vtkXMLUnstructuredGridWriter>::New();
//        writer->SetFileName(oss.str().c_str());
//        writer->SetInputData(mesh);
//        writer->SetDataModeToBinary();
//        writer->SetCompressionLevel(compressionLevel);
//
//#if VTK_MAJOR_VERSION >= 9
//        auto compressor = vtkSmartPointer<vtkZLibDataCompressor>::New();
//        writer->SetCompressor(compressor);
//#else
//        writer->SetCompressorTypeToZLib();
//#endif
//
//        if (!writer->Write()) {
//            std::cerr << "Failed to write high-quality mesh: " << oss.str() << std::endl;
//            return false;
//        }
//
//        std::cout << "✅ High-quality mesh saved: " << oss.str() << std::endl;
//        ++count;
//    }
//
//    std::cout << "Total high-quality meshes saved: " << count << std::endl;
//    return true;
//}



}  // namespace medsolver
