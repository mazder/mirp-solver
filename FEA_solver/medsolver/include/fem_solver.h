#ifndef MEDSOLVER_FEM_SOLVER_H
#define MEDSOLVER_FEM_SOLVER_H

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <vector>
#include <string>

#include "mesh.h"

namespace medsolver {

class FEMSolver {
public:
    void SetMesh(const Mesh& mesh);
    void SetMaterial(double youngsModulus, double poissonRatio);

    void ApplyBoundaryConditions();
    void AssembleSystem();
    void Solve();
    void ExportDisplacementField(const std::string& filename);

private:
    std::vector<Node> nodes_;
    std::vector<Element> elements_;

    double E_ = 0.0;
    double nu_ = 0.0;

    Eigen::SparseMatrix<double> stiffness_;

    Eigen::VectorXd force_;
    Eigen::VectorXd displacement_;

    int numDofs_ = 0;

    void AssembleElementStiffness(const Element& elem, Eigen::MatrixXd& Ke);
};

}  // namespace medsolver

#endif  // MEDSOLVER_FEM_SOLVER_H
