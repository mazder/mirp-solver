#pragma once
#include <string>
#include <vector>
#include <vtkSmartPointer.h>
#include <vtkImageData.h>
#include <vtkUnstructuredGrid.h>


namespace medsolver {

class MeshGenerator {
public:
    MeshGenerator();
    ~MeshGenerator();
    bool LoadImageMask(const std::string& imPath, const std::string& mskPath);
    // Create mesh
    bool ConvertNiiImageMaskToMesh();

    bool ConvertMaskToTetraMeshesHighQuality(
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
    );

    bool ConvertMaskToTetraMeshesHighQuality2(
    bool applySmoothing,
    double smoothRelaxation,
    bool useFeatureEdgeSmoothing,
    double featureAngle,
    int subdivisionIterations,
    double tetQuality,
    bool applyDecimation,
    double targetReduction,
    double minVolume
    );

    // Save generated mesh to a file (format example: "vtk", "vtp", "vtu", etc.)
    bool SaveMesh(const std::string& filename, const std::string& format, int compressionLevel = 5);
private:
    // helper functions
    bool LoadStoreImageMask(const std::string& filename, bool imgf, bool mskf);
    // Store all loaded images and maskes
    std::vector<vtkSmartPointer<vtkImageData>> images_;
    std::vector<vtkSmartPointer<vtkImageData>> masks_;
    std::vector<vtkSmartPointer<vtkUnstructuredGrid>> meshes_;

};

}  // namespace medsolver

