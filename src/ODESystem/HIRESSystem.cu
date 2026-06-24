// HIRESSystem.cpp
#include "HIRESSystem.cuh"

__global__ void 
constructGPU(kodes::HIRESSystem* system, const label sizeOfSystem)
{
    new (system) kodes::HIRESSystem(sizeOfSystem);
}

__global__ void 
destructGPU(kodes::HIRESSystem* system) {
    delete system;
}

__host__  kodes::HIRESSystem* 
kodes::HIRESSystem::createGPU(const label sizeOfSystem) {
    HIRESSystem* ptr;
    cudaMalloc(&ptr, sizeof(HIRESSystem));
    constructGPU<<<1, 1>>>(ptr, sizeOfSystem);
    cudaDeviceSynchronize();
    return ptr;
}

__host__  void
kodes::HIRESSystem::destroyGPU(kodes::HIRESSystem* system) {
    if (system) {
        destructGPU<<<1, 1>>>(system);
        cudaDeviceSynchronize();
        cudaFree(system);
    }
}

__device__
void kodes::HIRESSystem::derivatives(const scalar x, const scalar* y, scalar* dydx) const
{
    scalar y1 = y[0];
    scalar y2 = y[1];
    scalar y3 = y[2];
    scalar y4 = y[3];
    scalar y5 = y[4];
    scalar y6 = y[5];
    scalar y7 = y[6];
    scalar y8 = y[7];
    
    dydx[0] = -1.71 * y1 + 0.43 * y2 + 8.32 * y3 + 0.0007;
    dydx[1] = 1.71 * y1 - 8.75 * y2;
    dydx[2] = -10.03 * y3 + 0.43 * y4 + 0.035 * y5;
    dydx[3] = 8.32 * y2 + 1.71 * y3 - 1.12 * y4;
    dydx[4] = -1.745 * y5 + 0.43 * y6 + 0.43 * y7;
    dydx[5] = -280.0 * y6 * y8 + 0.69 * y4 + 1.71 * y5 - 0.43 * y6 + 0.69 * y7;
    dydx[6] = 280.0 * y6 * y8 - 1.81 * y7;
    dydx[7] = -280 * y6 * y8 + 1.81 * y7;
}

__device__
void kodes::HIRESSystem::jacobian(const scalar x, const scalar* y, scalar* dfdx, scalar* dfdy) const
{
    label sizeOfSystem_ = 8;
    
    // df/dx = 0 for autonomous system
    for (label i = 0; i < sizeOfSystem_; ++i)
    {
        dfdx[i] = 0.0;
    }
    
    // Initialize Jacobian matrix with zeros
    for (label i = 0; i < sizeOfSystem_; ++i)
    {
        for (label j = 0; j < sizeOfSystem_; ++j)
        {
            dfdy[i*sizeOfSystem_ + j] = 0.0;
        }
    }
    
    scalar y6 = y[5];
    scalar y8 = y[7];
    
    // Row 0: derivatives of y1'
    dfdy[0*sizeOfSystem_ + 0] = -1.71;
    dfdy[0*sizeOfSystem_ + 1] = 0.43;
    dfdy[0*sizeOfSystem_ + 2] = 8.32;

    // Row 1: derivatives of y2'
    dfdy[1*sizeOfSystem_ + 0] = 1.71;
    dfdy[1*sizeOfSystem_ + 1] = -8.75;

    // Row 2: derivatives of y3'
    dfdy[2*sizeOfSystem_ + 2] = -10.03;
    dfdy[2*sizeOfSystem_ + 3] = 0.43;
    dfdy[2*sizeOfSystem_ + 4] = 0.035;

    // Row 3: derivatives of y4'
    dfdy[3*sizeOfSystem_ + 1] = 8.32;
    dfdy[3*sizeOfSystem_ + 2] = 1.71;
    dfdy[3*sizeOfSystem_ + 3] = -1.12;

    // Row 4: derivatives of y5'
    dfdy[4*sizeOfSystem_ + 4] = -1.745;
    dfdy[4*sizeOfSystem_ + 5] = 0.43;
    dfdy[4*sizeOfSystem_ + 6] = 0.43;

    // Row 5: derivatives of y6'
    dfdy[5*sizeOfSystem_ + 3] = 0.69;
    dfdy[5*sizeOfSystem_ + 4] = 1.71;
    dfdy[5*sizeOfSystem_ + 5] = -280.0 * y8 - 0.43;
    dfdy[5*sizeOfSystem_ + 6] = 0.69;
    dfdy[5*sizeOfSystem_ + 7] = -280.0 * y6;

    // Row 6: derivatives of y7'
    dfdy[6*sizeOfSystem_ + 5] = 280.0 * y8;
    dfdy[6*sizeOfSystem_ + 6] = -1.81;
    dfdy[6*sizeOfSystem_ + 7] = 280.0 * y6;

    // Row 7: derivatives of y8'
    dfdy[7*sizeOfSystem_ + 5] = -280 * y8;
    dfdy[7*sizeOfSystem_ + 6] = 1.81;
    dfdy[7*sizeOfSystem_ + 7] = -280 * y6;
}
