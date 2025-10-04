#include <mpi.h>
#include <iostream>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);  // Initialize MPI

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // Get the rank of the process

    std::cout << "Hello from rank " << rank << std::endl;

    MPI_Finalize();  // Finalize MPI
    return 0;
}
