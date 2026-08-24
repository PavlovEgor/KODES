#include "pyJacSystem.cuh"


__global__ void 
constructGPU(kodes::pyJacSystem* system, mechanism_memory *d_mem)
{
    new (system) kodes::pyJacSystem(d_mem);
}

__global__ void 
destructGPU(kodes::pyJacSystem* system) {
    system->~pyJacSystem();
}

__host__  kodes::pyJacSystem* 
kodes::pyJacSystem::createGPU(mechanism_memory *d_mem) {
    if (!d_mem)
    {
        fprintf(stderr, "pyJacSystem::createGPU error at %s:%d: d_mem is null\n", __FILE__, __LINE__);
        std::exit(EXIT_FAILURE);
    }

    pyJacSystem* ptr;
    CUDA_CHECK(cudaMalloc(&ptr, sizeof(pyJacSystem)));
    constructGPU<<<1, 1>>>(ptr, d_mem);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());
    return ptr;
}

__host__  void
kodes::pyJacSystem::destroyGPU(kodes::pyJacSystem* system) {
    if (system) {
        destructGPU<<<1, 1>>>(system);
        CUDA_CHECK_LAST();
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(system));
    }
}

__device__
void kodes::pyJacSystem::derivatives(const scalar t, const scalar pressure, const scalar* y, scalar* dy) const
{
    dydt(t, pressure, y, dy, device_memory);
}

__device__
void kodes::pyJacSystem::jacobian(const scalar t, const scalar pressure, const scalar* y, scalar* dfdt, scalar* dfdy) const
{
    eval_jacob(t, pressure, y, dfdy, device_memory);
}
