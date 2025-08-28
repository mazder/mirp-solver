#include <iostream>
#include <cgnslib.h>
#include <vector>
#include <string>
#include <fstream>
#include "json.hpp"

#define FILENAME "example.cgns"

// Namespace for nlohmann JSON
using json = nlohmann::json;

// Class to represent the grid
class Grid {
public:
    std::vector<std::vector<int>> transfinite_vertices;
    std::vector<double> x_coords, y_coords;

    // Constructor to initialize the grid from a JSON object
    Grid(const json& json_data) {
        // Read grid data from JSON
        int imax = json_data["grid"]["imax"];
        int jmax = json_data["grid"]["jmax"];

        // Resize grid to match imax and jmax
        transfinite_vertices.resize(imax, std::vector<int>(jmax, 1)); // Dummy data for vertices

        // Check for "coordinates" and their validity
        const auto& x_array = json_data["coordinates"]["x"];
        const auto& y_array = json_data["coordinates"]["y"];

        // Ensure the coordinates arrays are valid and the sizes match imax * jmax
        if (x_array.size() != imax * jmax || y_array.size() != imax * jmax) {
            std::cerr << "Error: Coordinates size mismatch!" << std::endl;
            throw std::runtime_error("Mismatch in coordinates size.");
        }

        // Read vertex coordinates from JSON
        for (size_t i = 0; i < x_array.size(); ++i) {
            x_coords.push_back(x_array[i]);
            y_coords.push_back(y_array[i]);
        }

        // Debugging print to verify the coordinates
        std::cout << "Read " << x_coords.size() << " x-coordinates and " << y_coords.size() << " y-coordinates." << std::endl;
    }

    std::string getZoneName() const {
        return "Zone1";  // Example zone name
    }
};

int main() {
    // Step 1: Read JSON File
    std::ifstream json_file("../data.json");
    if (!json_file.is_open()) {
        std::cerr << "Error opening JSON file." << std::endl;
        return -1;
    }

    json json_data;
    json_file >> json_data;

    if (json_data.contains("grid")) {
        std::cout << "Grid data: " << json_data["grid"].dump(4) << std::endl;
    } else {
        std::cerr << "Grid data is missing!" << std::endl;
        return -1;
    }

    // Step 2: Create Grid from JSON data
    Grid gf(json_data);

    cgsize_t imax = gf.transfinite_vertices.size();
    cgsize_t jmax = gf.transfinite_vertices[0].size();

    // Size array to be passed to cg_zone_write
    cgsize_t cgZoneSize[6] = {imax, jmax, imax - 1, jmax - 1, 0, 0};  // [vertices_x, vertices_y, cells_x, cells_y, cells_z, empty]

    // Step 3: Open the CGNS file
    int cgIndexFile;
    if (cg_open(FILENAME, CG_MODE_WRITE, &cgIndexFile) != CG_OK) {
        std::cerr << "Error opening file: " << cg_get_error() << std::endl;
        return -1;
    }

    // Step 4: Write the base
    int cgIndexBase;
    if (cg_base_write(cgIndexFile, "Base", 2, 2, &cgIndexBase) != CG_OK) {
        std::cerr << "Error creating base: " << cg_get_error() << std::endl;
        cg_close(cgIndexFile);
        return -1;
    }

    // Step 5: Write the zone
    int cgIndexZone;
    std::string zoneName = gf.getZoneName();
    if (cg_zone_write(cgIndexFile, cgIndexBase, zoneName.c_str(), cgZoneSize, CGNS_ENUMV(Structured), &cgIndexZone) != CG_OK) {
        std::cerr << "Error creating zone: " << cg_get_error() << std::endl;
        cg_close(cgIndexFile);
        return -1;
    }

    // Step 6: Write the vertex coordinates to the CGNS file
    // Write x coordinates
    int cgIndexX;
    cgsize_t x_dim[1] = {static_cast<cgsize_t>(gf.x_coords.size())}; // Dimension for x array
    if (cg_array_write("CoordinateX", CGNS_ENUMV(RealDouble), 1, x_dim, gf.x_coords.data()) != CG_OK) {
        std::cerr << "Error writing X coordinates: " << cg_get_error() << std::endl;
        cg_close(cgIndexFile);
        return -1;
    }

    // Write y coordinates
    int cgIndexY;
    cgsize_t y_dim[1] = {static_cast<cgsize_t>(gf.y_coords.size())}; // Dimension for y array
    if (cg_array_write("CoordinateY", CGNS_ENUMV(RealDouble), 1, y_dim, gf.y_coords.data()) != CG_OK) {
        std::cerr << "Error writing Y coordinates: " << cg_get_error() << std::endl;
        cg_close(cgIndexFile);
        return -1;
    }

    // Step 7: Close the file
    if (cg_close(cgIndexFile) != CG_OK) {
        std::cerr << "Error closing file: " << cg_get_error() << std::endl;
        return -1;
    }

    std::cout << "CGNS file '" << FILENAME << "' created successfully." << std::endl;
    return 0;
}
