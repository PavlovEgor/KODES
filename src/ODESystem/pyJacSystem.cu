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
    pyJacSystem* ptr;
    cudaMalloc(&ptr, sizeof(pyJacSystem));
    constructGPU<<<1, 1>>>(ptr, d_mem);
    cudaDeviceSynchronize();
    return ptr;
}

__host__  void
kodes::pyJacSystem::destroyGPU(kodes::pyJacSystem* system) {
    if (system) {
        destructGPU<<<1, 1>>>(system);
        cudaDeviceSynchronize();
        cudaFree(system);
    }
}

__device__
void kodes::pyJacSystem::derivatives(const scalar x, const scalar pressure, const scalar* y, scalar* dydx) const
{
    dydt(x, pressure, y, dydx, device_memory);
}

__device__
void kodes::pyJacSystem::jacobian(const scalar x, const scalar pressure, const scalar* y, scalar* dfdx, scalar* dfdy) const
{
    eval_jacob(x, pressure, y, dfdy, device_memory);
}
