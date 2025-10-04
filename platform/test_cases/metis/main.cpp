#define IDXTYPEWIDTH 64
#define REALTYPEWIDTH 64
#include <metis.h>
#include <iostream>

int main() {
    // Simple graph with 2 vertices connected to each other
    idx_t nvtxs = 2;               // number of vertices
    idx_t ncon = 1;                // number of balancing constraints
    idx_t xadj[3] = {0, 1, 2};     // adjacency structure
    idx_t adjncy[2] = {1, 0};      // adjacency structure
    idx_t nparts = 2;              // number of parts
    idx_t part[2];                 // output partition

    // Optional arguments set to nullptr
    idx_t *vwgt = nullptr;
    idx_t *vsize = nullptr;
    idx_t *adjwgt = nullptr;
    real_t *tpwgts = nullptr;
    real_t *ubvec = nullptr;
    idx_t *options = nullptr;
    idx_t objval;

    int status = METIS_PartGraphKway(
        &nvtxs, &ncon, xadj, adjncy,
        vwgt, vsize, adjwgt,
        &nparts, tpwgts, ubvec,
        options, &objval, part
    );

    std::cout << "METIS status: " << status << std::endl;
    std::cout << "Partition: " << part[0] << ", " << part[1] << std::endl;
}
