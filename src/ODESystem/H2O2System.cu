#include "H2O2System.cuh"


__global__ void 
constructGPU(kodes::H2O2System* system, mechanism_memory *d_mem)
{
    new (system) kodes::H2O2System(d_mem);
}

__global__ void 
destructGPU(kodes::H2O2System* system) {
    system->~H2O2System();
}

__host__  kodes::H2O2System* 
kodes::H2O2System::createGPU(mechanism_memory *d_mem) {
    H2O2System* ptr;
    cudaMalloc(&ptr, sizeof(H2O2System));
    constructGPU<<<1, 1>>>(ptr, d_mem);
    cudaDeviceSynchronize();
    return ptr;
}

__host__  void
kodes::H2O2System::destroyGPU(kodes::H2O2System* system) {
    if (system) {
        destructGPU<<<1, 1>>>(system);
        cudaDeviceSynchronize();
        cudaFree(system);
    }
}

__device__
void kodes::H2O2System::derivatives(const scalar x, const scalar rho, const scalar* y, scalar* dydx) const
{
    dydt(x, rho, y, dydx, device_memory);
}

__device__
void kodes::H2O2System::jacobian(const scalar x, const scalar rho, const scalar* y, scalar* dfdx, scalar* dfdy) const
{
    eval_jacob(x, rho, y, dfdy, device_memory);
}
