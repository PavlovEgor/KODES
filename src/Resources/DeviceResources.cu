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
    CUDA_CHECK(cudaMalloc(&ptr, sizeof(DeviceResources)));

    constructDeviceResources<<<1, 1>>>(ptr, batchSize, systemSize, parameterSize);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMalloc(&ptr->vectors, systemSize * batchSize * sizeof(scalar)));
    CUDA_CHECK(cudaMalloc(&ptr->parameters, parameterSize * batchSize * sizeof(scalar)));

    return ptr;
}

__host__  void
kodes::DeviceResources::destroy(kodes::DeviceResources* devRes) {
    if (devRes) {

        CUDA_CHECK(cudaFree(devRes->vectors));
        CUDA_CHECK(cudaFree(devRes->parameters));

        destructDeviceResources<<<1, 1>>>(devRes);
        CUDA_CHECK_LAST();
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaFree(devRes));
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
