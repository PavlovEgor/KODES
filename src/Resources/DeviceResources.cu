#include "DeviceResources.cuh"

namespace kodes 
{

__global__ void 
constructDeviceResources(kodes::DeviceResources* devRes, const label batchSize, const label systemSize, const label parameterSize)
{
    new (devRes) kodes::DeviceResources(batchSize, systemSize, parameterSize);
}

__global__ void 
destructDeviceResources(kodes::DeviceResources* devRes) {
    delete devRes;
}

__host__  kodes::DeviceResources* 
kodes::DeviceResources::create(const label batchSize, const label systemSize, const label parameterSize) {
    DeviceResources* ptr;
    cudaMalloc(&ptr, sizeof(DeviceResources));
    constructDeviceResources<<<1, 1>>>(ptr, batchSize, systemSize, parameterSize);
    cudaDeviceSynchronize();

    cudaMalloc(&ptr->vectors, systemSize * batchSize * sizeof(scalar));
    cudaMalloc(&ptr->parameters, parameterSize * batchSize * sizeof(scalar));

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
    for (label j = 0; j < systemSize_; ++j) {
        printf("%0.2f ", this->vectors[(j)]);
    }
    printf("\n");
}

}
