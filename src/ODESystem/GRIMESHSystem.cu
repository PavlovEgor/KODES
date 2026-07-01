// GRIMESHSystem.cpp
#include "GRIMESHSystem.cuh"


__global__ void 
constructGPU(kodes::GRIMESHSystem* system, mechanism_memory *d_mem)
{
    new (system) kodes::GRIMESHSystem(d_mem);
}

__global__ void 
destructGPU(kodes::GRIMESHSystem* system) {
    delete system;
}

__host__  kodes::GRIMESHSystem* 
kodes::GRIMESHSystem::createGPU(mechanism_memory *d_mem) {
    GRIMESHSystem* ptr;
    cudaMalloc(&ptr, sizeof(GRIMESHSystem));
    constructGPU<<<1, 1>>>(ptr, d_mem);
    cudaDeviceSynchronize();
    return ptr;
}

__host__  void
kodes::GRIMESHSystem::destroyGPU(kodes::GRIMESHSystem* system) {
    if (system) {
        destructGPU<<<1, 1>>>(system);
        cudaDeviceSynchronize();
        cudaFree(system);
    }
}

__device__
void kodes::GRIMESHSystem::derivatives(const scalar x, const scalar* y, scalar* dydx) const
{
    dydt(x, 101325.0, y, dydx, device_memory);
}

__device__
void kodes::GRIMESHSystem::jacobian(const scalar x, const scalar* y, scalar* dfdx, scalar* dfdy) const
{
    eval_jacob(x, 101325.0, y, dfdy, device_memory);
}
