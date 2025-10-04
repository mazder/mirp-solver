#include "mesh.h"

#include <vtkSmartPointer.h>
#include <vtkUnstructuredGridReader.h>
#include <vtkXMLUnstructuredGridReader.h>
#include <vtkXMLUnstructuredGridWriter.h>
#include <vtkUnstructuredGrid.h>
#include <vtkPoints.h>
#include <vtkCellData.h>
#include <vtkCellType.h>
#include <vtkCell.h>
#include <vtkIntArray.h>
#include <iostream>
#include <vtkTetra.h>
#include <vtkCellArray.h>


namespace medsolver {

bool Mesh::LoadFromFile(const std::string& filename, bool convertVoxels) {
    if (filename.ends_with(".vtu")) {
        auto reader = vtkSmartPointer<vtkXMLUnstructuredGridReader>::New();
        reader->SetFileName(filename.c_str());
        reader->Update();
        return processReaderOutput(reader->GetOutput(), filename, convertVoxels);
    } else {
        auto reader = vtkSmartPointer<vtkUnstructuredGridReader>::New();
        reader->SetFileName(filename.c_str());
        reader->Update();
        return processReaderOutput(reader->GetOutput(), filename, convertVoxels);
    }
}


// Shared logic for populating mesh structures
bool Mesh::processReaderOutput(vtkUnstructuredGrid* grid, const std::string& filename, bool useVoxelToTetrahedra) {
    if (!grid) {
        std::cerr << "Failed to read mesh file: " << filename << std::endl;
        return false;
    }

    auto points = grid->GetPoints();
    int numPoints = points->GetNumberOfPoints();
    int numCells = grid->GetNumberOfCells();

    nodes_.clear();
    nodes_.reserve(numPoints);
    for (int i = 0; i < numPoints; ++i) {
        double p[3]; points->GetPoint(i, p);
        nodes_.emplace_back(Node{p[0], p[1], p[2]});
    }

    elements_.clear();

    auto regionArray = vtkIntArray::SafeDownCast(grid->GetCellData()->GetArray("RegionId"));

    int numTetrahedra = 0;
    int numFromVoxels = 0;

    for (int i = 0; i < numCells; ++i) {
        auto cellType = grid->GetCellType(i);

        if (cellType == VTK_TETRA) {
            numTetrahedra++;
            auto cell = grid->GetCell(i);
            std::array<int, 4> ids;
            for (int j = 0; j < 4; ++j)
                ids[j] = cell->GetPointId(j);

            int region_id = regionArray ? regionArray->GetValue(i) : 0;
            elements_.emplace_back(Element{ids, region_id});
        }
        else if (cellType == VTK_VOXEL && useVoxelToTetrahedra) {
            numFromVoxels+=5;
            auto cell = grid->GetCell(i);
            std::array<int, 8> vox;
            for (int j = 0; j < 8; ++j)
                vox[j] = cell->GetPointId(j);

            // Split voxel into 5 tetrahedra
            const std::array<std::array<int, 4>, 5> voxelToTets = {{
                {vox[0], vox[1], vox[2], vox[4]},
                {vox[1], vox[2], vox[5], vox[6]},
                {vox[2], vox[4], vox[6], vox[7]},
                {vox[0], vox[2], vox[3], vox[7]},
                {vox[0], vox[4], vox[2], vox[7]}
            }};

            int region_id = regionArray ? regionArray->GetValue(i) : 0;
            for (const auto& tet : voxelToTets)
                elements_.emplace_back(Element{tet, region_id});
        }
        else {
            std::cerr << "Skipping unsupported cell type: " << cellType << std::endl;
        }
    }

    std::cout << "Loaded mesh " << filename << " with " << nodes_.size() << " nodes and:\n";
    if (numTetrahedra > 0)
        std::cout << "  - " << numTetrahedra << " tetrahedral elements\n";
    if (numFromVoxels > 0)
        std::cout << "  - " << numFromVoxels << " converted from voxels\n";
    if (numTetrahedra == 0 && numFromVoxels == 0)
        std::cout << "  - No usable elements found (only unsupported cell types?)\n";

    return true;
}

bool Mesh::WriteToVTU(const std::string& filename) const {
    auto points = vtkSmartPointer<vtkPoints>::New();
    for (const auto& node : nodes_)
        points->InsertNextPoint(node.x, node.y, node.z);

    auto grid = vtkSmartPointer<vtkUnstructuredGrid>::New();
    grid->SetPoints(points);

    auto regionArray = vtkSmartPointer<vtkIntArray>::New();
    regionArray->SetName("RegionId");

    for (const auto& elem : elements_) {
        auto tetra = vtkSmartPointer<vtkTetra>::New();
        for (int i = 0; i < 4; ++i)
            tetra->GetPointIds()->SetId(i, elem.node_ids[i]);

        grid->InsertNextCell(tetra->GetCellType(), tetra->GetPointIds());
        regionArray->InsertNextValue(elem.region_id);
    }

    grid->GetCellData()->AddArray(regionArray);

    auto writer = vtkSmartPointer<vtkXMLUnstructuredGridWriter>::New();
    writer->SetFileName(filename.c_str());
    writer->SetInputData(grid);
    writer->Write();

    std::cout << "Wrote mesh to " << filename << std::endl;
    return true;
}


} // namespace medsolver

//#include "mesh.h"
//
//#include <vtkSmartPointer.h>
//#include <vtkUnstructuredGridReader.h>
//#include <vtkUnstructuredGrid.h>
//#include <vtkPoints.h>
//#include <vtkCellArray.h>
//#include <vtkCellType.h>
//#include <vtkIntArray.h>
//#include <vtkCellData.h>
//#include <vtkCell.h>
//
//#include <iostream>
//
//namespace medsolver {
//
//bool Mesh::LoadFromVTK(const std::string& filename) {
//    auto reader = vtkSmartPointer<vtkUnstructuredGridReader>::New();
//    reader->SetFileName(filename.c_str());
//    reader->Update();
//
//    auto grid = reader->GetOutput();
//    if (!grid) {
//        std::cerr << "Failed to read VTK file: " << filename << std::endl;
//        return false;
//    }
//
//    auto points = grid->GetPoints();
//    int numPoints = points->GetNumberOfPoints();
//    int numCells = grid->GetNumberOfCells();
//
//    nodes_.clear();
//    nodes_.reserve(numPoints);
//
//    for (int i = 0; i < numPoints; ++i) {
//        double p[3];
//        points->GetPoint(i, p);
//        nodes_.emplace_back(Node{p[0], p[1], p[2]});
//    }
//
//    elements_.clear();
//    elements_.reserve(numCells);
//
//    auto regionArray = vtkIntArray::SafeDownCast(grid->GetCellData()->GetArray("RegionId"));
//    for (int i = 0; i < numCells; ++i) {
//        if (grid->GetCellType(i) != VTK_TETRA)
//            continue;
//
//        auto cell = grid->GetCell(i);
//        std::array<int, 4> ids;
//        for (int j = 0; j < 4; ++j)
//            ids[j] = cell->GetPointId(j);
//
//        int region_id = regionArray ? regionArray->GetValue(i) : 0;
//        elements_.emplace_back(Element{ids, region_id});
//    }
//
//    return true;
//}
//
//void Mesh::PrintInfo() const {
//    std::cout << "Mesh has " << nodes_.size() << " nodes, "
//              << elements_.size() << " tetrahedral elements." << std::endl;
//}
//
//}  // namespace medsolver
