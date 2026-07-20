#include "DeviceResources.cuh"

namespace kodes 
{

__global__ void 
constructDeviceResources(kodes::DeviceResources* devRes, const label numOfSystems, const label sizeOfSystem, const label numOfParameters)
{
    new (devRes) kodes::DeviceResources(numOfSystems, sizeOfSystem, numOfParameters);
}

__global__ void 
destructDeviceResources(kodes::DeviceResources* devRes) {
    delete devRes;
}

__host__  kodes::DeviceResources* 
kodes::DeviceResources::create(const label numOfSystems, const label sizeOfSystem, const label numOfParameters) {
    DeviceResources* ptr;
    cudaMalloc(&ptr, sizeof(DeviceResources));
    constructDeviceResources<<<1, 1>>>(ptr, numOfSystems, sizeOfSystem, numOfParameters);
    cudaDeviceSynchronize();

    cudaMalloc(&ptr->vectors, sizeOfSystem * numOfSystems * sizeof(scalar));
    cudaMalloc(&ptr->parameters, numOfParameters * numOfSystems * sizeof(scalar));

    return ptr;
}

__host__  void
kodes::DeviceResources::destroy(kodes::DeviceResources* devRes) {
    if (devRes) {

        cudaFree(devRes->vectors);
        cudaFree(devRes->parameters);

        destructDeviceResources<<<1, 1>>>(devRes);
        cudaDeviceSynchronize();
        cudaFree(devRes);
    }
}

__host__ __device__ void 
DeviceResources::printVectori(const label i) const
{
    for (label j = 0; j < sizeOfSystem_; ++j) {
        printf("%0.2f ", this->vectors[(j)]);
    }
    printf("\n");
}

}
