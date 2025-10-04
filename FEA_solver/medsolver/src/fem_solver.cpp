#include "fem_solver.h"
#include <iostream>

namespace medsolver {

void FEMSolver::SetMesh(const Mesh& mesh) {
    nodes_ = mesh.GetNodes();
    elements_ = mesh.GetElements();
    numDofs_ = static_cast<int>(nodes_.size()) * 3;

    if (numDofs_ <= 0) {
        std::cerr << "[Error] Mesh contains no nodes!" << std::endl;
    }
}
void FEMSolver::SetMaterial(double youngsModulus, double poissonRatio) {
    E_ = youngsModulus;
    nu_ = poissonRatio;
}

void FEMSolver::ApplyBoundaryConditions() {
    std::cerr << "[Warning] ApplyBoundaryConditions not implemented.\n";
}

void FEMSolver::AssembleElementStiffness(const Element& /*elem*/, Eigen::MatrixXd& Ke) {
    Ke.resize(12, 12);
    Ke.setZero();
    // Dummy stiffness matrix: identity scaled by 1.0
    for (int i = 0; i < 12; ++i) {
        Ke(i, i) = 1.0;
    }
}

void FEMSolver::AssembleSystem() {
    if (numDofs_ <= 0) {
        throw std::runtime_error("Cannot assemble system: numDofs_ is not initialized.");
    }

    // Resize system matrices/vectors
    force_ = Eigen::VectorXd::Zero(numDofs_);
    stiffness_.resize(numDofs_, numDofs_);

    // Optionally reserve some space for performance
    stiffness_.reserve(Eigen::VectorXi::Constant(numDofs_, 60));  // Adjust 60 to avg nonzeros per row

    for (const auto& elem : elements_) {
        Eigen::MatrixXd Ke(12, 12);
        AssembleElementStiffness(elem, Ke);  // Should fill Ke with the element stiffness matrix

        for (int a = 0; a < 4; ++a) {
            int ni = elem.node_ids[a];
            for (int b = 0; b < 4; ++b) {
                int nj = elem.node_ids[b];

                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j) {
                        int row = 3 * ni + i;
                        int col = 3 * nj + j;
                        double val = Ke(3 * a + i, 3 * b + j);

                        // Add value to sparse matrix directly
                        stiffness_.coeffRef(row, col) += val;
                    }
                }
            }
        }
    }

    stiffness_.makeCompressed();  // Optional: prepares matrix for solver
    std::cout << "[Info] Assembled system with " << numDofs_ << " DOFs.\n";
}

void FEMSolver::Solve() {

    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;

    solver.analyzePattern(stiffness_);
    solver.factorize(stiffness_);
    displacement_ = solver.solve(force_);

    std::cout << "[Info] Linear system solved. Norm of displacement = " << displacement_.norm() << std::endl;
}

void FEMSolver::ExportDisplacementField(const std::string& /*filename*/) {
    std::cerr << "[Warning] ExportDisplacementField not implemented.\n";
}

}  // namespace medsolver
