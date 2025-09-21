#include <mpi.h>
#include "mfem.hpp"
//#include "core/mirp_sim.hpp"
//#include "config/cgns_interface.hpp"
//#include "core/device_manager.hpp"

int main(int argc, char *argv[])
{
    // ───────────────────────────────────────────────
    // 1. MPI & GPU Initialization
    // ───────────────────────────────────────────────
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

//    // Set device for this rank (CPU or GPU)
//    DeviceManager::SetDeviceByRank(rank);  // Custom logic to assign CUDA device per rank
//
//    // Configure MFEM device backend (cuda, cpu, hip, etc.)
//    mfem::Device device("cuda"); // or "cpu", "hip", etc.
//    mfem::Device::Configure(device);

    if (rank == 0)
        std::cout << "[MIRP_SIM] Starting simulation..." << std::endl;

//    // ───────────────────────────────────────────────
//    // 2. Read Simulation Configuration from CGNS
//    // ───────────────────────────────────────────────
//    std::string input_file = "input.cgns";
//    SimulationConfig config = CGNSInterface::Load(input_file);
//
//    // ───────────────────────────────────────────────
//    // 3. Run Simulation
//    // ───────────────────────────────────────────────
//    MirpSim sim(config);
//    sim.Run();

    // ───────────────────────────────────────────────
    // 4. Finalize MPI
    // ───────────────────────────────────────────────
    MPI_Finalize();
    return 0;
}
