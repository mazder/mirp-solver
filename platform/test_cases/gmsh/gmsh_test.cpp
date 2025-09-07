#include <gmsh.h>
#include <iostream>

int main(int argc, char **argv) {
    try {
        gmsh::initialize();
        gmsh::model::add("tumor_mesh");

        // Example: add a point
        double lc = 1.0;
        gmsh::model::geo::addPoint(0, 0, 0, lc, 1);
        gmsh::model::geo::addPoint(1, 0, 0, lc, 2);
        gmsh::model::geo::addPoint(0, 1, 0, lc, 3);

        // Example: create line loop and surface
        gmsh::model::geo::addLine(1, 2, 1);
        gmsh::model::geo::addLine(2, 3, 2);
        gmsh::model::geo::addLine(3, 1, 3);
        gmsh::model::geo::addCurveLoop({1, 2, 3}, 1);
        gmsh::model::geo::addPlaneSurface({1}, 1);

        gmsh::model::geo::synchronize();

        gmsh::model::mesh::generate(2);
        gmsh::write("example.msh");

        gmsh::finalize();
        std::cout << "✅ Mesh saved as example.msh\n";
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        gmsh::finalize();
        return 1;
    }
    return 0;
}
