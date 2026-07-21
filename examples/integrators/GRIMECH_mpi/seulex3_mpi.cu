#include "seulex3_mpi.cuh"

// MPI counterpart to GRIMECH/seulex3.cu: every rank solves its own, fully
// independent copy of the same 257-system GRIMech 3.0 problem for 10s, on its
// own bound CUDA device. There is no scatter/gather between ranks - per the
// MPI contract in kodes_mpi.cuh, each rank is assumed to already own its
// local slice of systems, so this just exercises kodes::mpiSelectDevice()
// plus the existing single-GPU solve path running concurrently across
// ranks/GPUs.
int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int worldRank = 0;
    int worldSize = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    const int device = kodes::mpiSelectDevice();

    label numOfSystems = 257;

    kodes::HostResources            host_res(numOfSystems, NSP, 1);

    set_same_initial_conditions(host_res.numOfSystems(), host_res.vectors, host_res.parameters);

    printf("[rank %d/%d] cuda device %d, solving %d systems\n", worldRank, worldSize, device, numOfSystems);
    host_res.printVectori(0);

    mechanism_memory *h_mem = (mechanism_memory*)malloc(sizeof(mechanism_memory));
    mechanism_memory *d_mem = nullptr;

    initialize_gpu_memory(kodes::paddedNumOfSystems(numOfSystems), &h_mem, &d_mem);

    kodes::SeulexDeviceResources   host_res_dev(host_res.numOfSystems(), host_res.sizeOfSystem(), host_res.numOfParameters());

    kodes::SeulexDeviceResources*   res_prt = kodes::SeulexDeviceResources::create(numOfSystems, host_res.sizeOfSystem(), 1, &host_res_dev);

    kodes::pyJacSystem* ode_prt = kodes::pyJacSystem::createGPU(d_mem);

    kodes::Operator<kodes::HostResources, kodes::SeulexDeviceResources> op(&host_res, &host_res_dev);

    kodes::Seulex<kodes::pyJacSystem> solver(ode_prt, res_prt, host_res.numOfSystems());

    op.cpyHostToDevice();

    scalar xEnd = 10.0;
    stepState step(xEnd);

    const auto t0 = std::chrono::steady_clock::now();

    solver.solve(step);

    op.cpyDeviceToHost();

    const auto t1 = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(t1 - t0).count();

    printf("[rank %d/%d] solved in %.3f s\n", worldRank, worldSize, elapsed);
    host_res.printVectori(0);

    kodes::pyJacSystem::destroyGPU(ode_prt);
    kodes::SeulexDeviceResources::destroy(res_prt, &host_res_dev);

    MPI_Finalize();

    return 0;
}
