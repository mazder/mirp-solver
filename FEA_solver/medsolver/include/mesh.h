#pragma once

#include <vector>
#include <array>
#include <string>

class vtkUnstructuredGrid;

namespace medsolver {

struct Node { double x, y, z; };
struct Element { std::array<int, 4> node_ids; int region_id; };

class Mesh {
public:
    bool LoadFromFile(const std::string& filename, bool convertVoxels = false);
    bool WriteToVTU(const std::string& filename) const;

    const std::vector<Node>& GetNodes() const { return nodes_; }
    const std::vector<Element>& GetElements() const { return elements_; }

private:
    bool processReaderOutput(vtkUnstructuredGrid* grid, const std::string& filename, bool useVoxelToTetrahedra = false);

    std::vector<Node> nodes_;
    std::vector<Element> elements_;
};

}  // namespace medsolver



//#pragma once
//
//#include <vector>
//#include <array>
//#include <string>
//
//// Forward declaration of VTK class
//class vtkUnstructuredGrid;
//
//namespace medsolver {
//
//struct Node {
//    double x, y, z;
//};
//
//struct Element {
//    std::array<int, 4> node_ids;
//    int region_id;
//};
//
//class Mesh {
//public:
//    bool LoadFromVTK(const std::string& filename);
//    void PrintInfo() const;
//
//    const std::vector<Node>& GetNodes() const { return nodes_; }
//    const std::vector<Element>& GetElements() const { return elements_; }
//
//private:
//    std::vector<Node> nodes_;
//    std::vector<Element> elements_;
//};
//
//}  // namespace medsolver
