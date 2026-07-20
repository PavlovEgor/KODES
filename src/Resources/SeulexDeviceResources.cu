#include "SeulexDeviceResources.cuh"


__global__ void 
constructSeulexDeviceResources(kodes::SeulexDeviceResources* devRes, const label numOfSystems, const label sizeOfSystem, const label numOfParameters)
{
    new (devRes) kodes::SeulexDeviceResources(numOfSystems, sizeOfSystem, numOfParameters);
}

__global__ void 
destructSeulexDeviceResources(kodes::SeulexDeviceResources* devRes) {
    devRes->~SeulexDeviceResources();
}

__host__  kodes::SeulexDeviceResources* 
kodes::SeulexDeviceResources::create(const label numOfSystems, const label sizeOfSystem, const label numOfParameters, kodes::SeulexDeviceResources* hostStub) {
    SeulexDeviceResources* devPtr;
    
    cudaMalloc(&devPtr, sizeof(SeulexDeviceResources));
    
    cudaMalloc(&hostStub->vectors, sizeOfSystem * numOfSystems * sizeof(scalar));
    cudaMalloc(&hostStub->parameters, numOfParameters * numOfSystems * sizeof(scalar));

    cudaMalloc(&hostStub->table_, 12 * sizeOfSystem * numOfSystems * sizeof(scalar));
    cudaMalloc(&hostStub->dfdx_, sizeOfSystem * numOfSystems * sizeof(scalar));
    cudaMalloc(&hostStub->dfdy_, sizeOfSystem * sizeOfSystem * numOfSystems * sizeof(scalar));
    cudaMalloc(&hostStub->a_, sizeOfSystem * sizeOfSystem * numOfSystems * sizeof(scalar));

    cudaMalloc(&hostStub->pivotIndices_, sizeOfSystem * numOfSystems * sizeof(label));

    cudaMalloc(&hostStub->dxOpt_, sizeOfSystem * numOfSystems * sizeof(scalar));
    cudaMalloc(&hostStub->temp_, sizeOfSystem * numOfSystems * sizeof(scalar));
    cudaMalloc(&hostStub->y0_, sizeOfSystem * numOfSystems * sizeof(scalar));
    cudaMalloc(&hostStub->ySequence_, sizeOfSystem * numOfSystems * sizeof(scalar));
    cudaMalloc(&hostStub->scale_, sizeOfSystem * numOfSystems * sizeof(scalar));
    cudaMalloc(&hostStub->dy_, sizeOfSystem * numOfSystems * sizeof(scalar));
    cudaMalloc(&hostStub->yTemp_, sizeOfSystem * numOfSystems * sizeof(scalar));
    cudaMalloc(&hostStub->dydx_, sizeOfSystem * numOfSystems * sizeof(scalar));
    cudaMalloc(&hostStub->y_, sizeOfSystem * numOfSystems * sizeof(scalar));
    
    cudaMemcpy(devPtr, hostStub, sizeof(SeulexDeviceResources), cudaMemcpyHostToDevice);
    
    constructSeulexDeviceResources<<<1, 1>>>(devPtr, numOfSystems, sizeOfSystem, numOfParameters);
    cudaDeviceSynchronize();
    
    return devPtr;
}

__host__  void
kodes::SeulexDeviceResources::destroy(kodes::SeulexDeviceResources* devRes, kodes::SeulexDeviceResources* hostStub) {
    if (hostStub) {

        cudaFree(hostStub->vectors);
        cudaFree(hostStub->parameters);

        cudaFree(hostStub->table_);
        cudaFree(hostStub->dfdx_);
        cudaFree(hostStub->dfdy_);
        cudaFree(hostStub->a_);

        cudaFree(hostStub->pivotIndices_);

        cudaFree(hostStub->dxOpt_);
        cudaFree(hostStub->temp_);
        cudaFree(hostStub->y0_);
        cudaFree(hostStub->ySequence_);
        cudaFree(hostStub->scale_);
        cudaFree(hostStub->dy_);
        cudaFree(hostStub->yTemp_);
        cudaFree(hostStub->dydx_);
        cudaFree(hostStub->y_);

        destructSeulexDeviceResources<<<1, 1>>>(devRes);
        cudaDeviceSynchronize();
        cudaFree(devRes);
    }
}
