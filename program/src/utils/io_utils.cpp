#include "utils/io_utils.hpp"
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <cgnslib.h>

namespace IOUtils {

using json = nlohmann::json;

json readConfigFromJSON(const std::string& jsonFilePath) {
    std::ifstream file(jsonFilePath);

    // Handle file opening errors
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open the file: " << jsonFilePath << std::endl;
        throw std::runtime_error("File not found or could not be opened");
    }

    // Check if the file is empty
    file.seekg(0, std::ios::end);
    if (file.tellg() == 0) {
        std::cerr << "Error: The file is empty: " << jsonFilePath << std::endl;
        throw std::runtime_error("File is empty");
    }

    // Read and parse JSON data
    file.seekg(0, std::ios::beg);
    json config;
    try {
        file >> config;
    } catch (const json::parse_error& e) {
        std::cerr << "Error: Failed to parse JSON from the file: " << e.what() << std::endl;
        throw;  // Rethrow the exception
    }

    return config;
}

void createBasicCGNS(const std::string& cgnsFilePath, const json& config) {
    double x[9][17][21], y[9][17][21], z[9][17][21];
    cgsize_t isize[3][3];
    int ni, nj, nk, i, j, k;
    int index_file, icelldim, iphysdim, index_base;
    int index_zone, index_coord_x, index_coord_y, index_coord_z; // Separate indices for each coordinate

    ni = 21;
    nj = 17;
    nk = 9;

    // Initialize the grid coordinates
    for (k = 0; k < nk; k++) {
        for (j = 0; j < nj; j++) {
            for (i = 0; i < ni; i++) {
                x[k][j][i] = i;
                y[k][j][i] = j;
                z[k][j][i] = k;
            }
        }
    }

    // Open CGNS file for writing
    if (cg_open(cgnsFilePath.c_str(), CG_MODE_WRITE, &index_file) != CG_OK) {
        std::cerr << "Error opening CGNS file: " << cgnsFilePath << std::endl;
        return;
    }

    icelldim = 3;
    iphysdim = 3;

    // Create base (assuming it is of dimension 3 for the example)
    if (cg_base_write(index_file, "Base", icelldim, iphysdim, &index_base) != CG_OK) {
        std::cerr << "Error writing base to CGNS file!" << std::endl;
        cg_close(index_file);
        return;
    }

    // Set grid sizes for vertices and cells
    isize[0][0] = 21;
    isize[0][1] = 17;
    isize[0][2] = 9;
    isize[1][0] = isize[0][0] - 1;
    isize[1][1] = isize[0][1] - 1;
    isize[1][2] = isize[0][2] - 1;
    isize[2][0] = 0; // Boundary vertex size
    isize[2][1] = 0;
    isize[2][2] = 0;

    // Write zone to CGNS file with a family name
    if (cg_zone_write(index_file, index_base, "Zone1", *isize, Structured, &index_zone) != CG_OK) {
        std::cerr << "Error writing zone to CGNS file! CGNS Error Code: "<< std::endl;
        cg_close(index_file);
        return;
    }

    std::cout << "Zone successfully written to CGNS file!" << std::endl;

    // Write coordinates separately for X, Y, and Z with data class
    cg_coord_write(index_file, index_base, index_zone, RealDouble, "CoordinateX", x, &index_coord_x);
    cg_coord_write(index_file, index_base, index_zone, RealDouble, "CoordinateY", y, &index_coord_y);
    cg_coord_write(index_file, index_base, index_zone, RealDouble, "CoordinateZ", z, &index_coord_z);

    std::cout << "Grid coordinates successfully written to CGNS file!" << std::endl;

    // Close CGNS file
    cg_close(index_file);

    std::cout << "Successfully written to CGNS file: " << cgnsFilePath << std::endl;
}

void writeJsonConfigToCGNS(const std::string& cgnsFilePath, const json& j) {
    // Open the CGNS file for reading and writing
    int cgnsFile;
    std::cout << cgnsFilePath << std::endl;

    if (cg_open(cgnsFilePath.c_str(), CG_MODE_MODIFY, &cgnsFile) != CG_OK) {
        std::cerr << "Error opening CGNS file!" << std::endl;
        return;
    }
    else{
        std::cout << "opening CGNS file success" << std::endl;
    }
    // Iterate through all bases and Zones in the CGNS file to make sure CGNS file is not corrupted
    int numBases;
    if (cg_nbases(cgnsFile, &numBases) != CG_OK) {
        std::cerr << "Error reading number of bases in CGNS file!" << std::endl;
        cg_close(cgnsFile);
        return;
    }
    for (int baseIndex = 1; baseIndex <= numBases; ++baseIndex) {
        // Get the number of zones in this base
        int numZones;
        if (cg_nzones(cgnsFile, baseIndex, &numZones) != CG_OK) {
            std::cerr << "Error reading number of zones in base " << baseIndex << std::endl;
            cg_close(cgnsFile);
            return;
        }
        // Iterate through all zones in the base
        for (int zoneIndex = 1; zoneIndex <= numZones; ++zoneIndex) {
            char zoneName[100];  // Buffer for zone name
            cgsize_t zoneSize[3][3];  // Array for storing zone dimensions (usually 3 for 3D grids)
            // Read the zone name and size
            if (cg_zone_read(cgnsFile, baseIndex, zoneIndex, zoneName, *zoneSize) != CG_OK) {
                std::cerr << "Error reading zone details!" << std::endl;
                return;
            }
            std::cout << zoneName << " Processing Zone ----------------------" << zoneIndex << " in Base " << baseIndex << std::endl;
        }
    }

  //  cg_gopath(cgnsFile, "/Base");
  //  cg_user_data_write("Data");

    cg_close(cgnsFile);  // Close the CGNS file after all operations
}

void writeJSONToCGNS(const std::string& cgnsFilePath, const std::string& jsonFilePath) {
    // Read the JSON configuration file
    json config = readConfigFromJSON(jsonFilePath);

   // std::cout << "Config JSON: " << config.dump(4) << std::endl;
    // create a basic CGNS file with Zone1
    createBasicCGNS(cgnsFilePath, config);
    // Write the configuration to the CGNS file
    std::cout << "cgnsFilePath: " << cgnsFilePath << std::endl;
    //writeJsonConfigToCGNS(cgnsFilePath, config);
}

} // namespace IOUtils
