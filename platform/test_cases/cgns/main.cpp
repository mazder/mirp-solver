#include <iostream>
#include <cgnslib.h>
#include <vector>
#include <string>

#define FILENAME "example.cgns"

// Assuming a class structure like the one from your code
class Grid {
public:
    std::vector<std::vector<int>> transfinite_vertices;

    Grid(int imax, int jmax) {
        // Initialize the grid with some dummy values
        transfinite_vertices.resize(imax, std::vector<int>(jmax, 1)); // example 1 for each point
    }

    std::string getZoneName() const {
        return "Zone1";  // Example zone name
    }
};

int main() {
    // Create a Grid instance
    Grid gf(4, 4);  // Let's assume a 4x4 grid

    cgsize_t imax = gf.transfinite_vertices.size();
    cgsize_t jmax = gf.transfinite_vertices[0].size();

    // Size array to be passed to cg_zone_write
    cgsize_t cgZoneSize[6] = {imax, jmax, imax - 1, jmax - 1, 0, 0};  // [vertices_x, vertices_y, cells_x, cells_y, cells_z, empty]

    // Open the CGNS file
    int cgIndexFile;
    if (cg_open(FILENAME, CG_MODE_WRITE, &cgIndexFile) != CG_OK) {
        std::cerr << "Error opening file" << std::endl;
        return -1;
    }

    // Write the base
    int cgIndexBase;
    if (cg_base_write(cgIndexFile, "Base", 2, 2, &cgIndexBase) != CG_OK) {
        std::cerr << "Error creating base: " << cg_get_error() << std::endl;
        cg_close(cgIndexFile);
        return -1;
    }

    // Write the zone
    int cgIndexZone;
    std::string zoneName = gf.getZoneName();
    if (cg_zone_write(cgIndexFile, cgIndexBase, zoneName.c_str(), cgZoneSize, CGNS_ENUMV(Structured), &cgIndexZone) != CG_OK) {
        std::cerr << "Error creating zone: " << cg_get_error() << std::endl;
        cg_close(cgIndexFile);
        return -1;
    }

    // Close the file
    if (cg_close(cgIndexFile) != CG_OK) {
        std::cerr << "Error closing file: " << cg_get_error() << std::endl;
        return -1;
    }

    std::cout << "CGNS file '" << FILENAME << "' created successfully." << std::endl;
    return 0;
}
