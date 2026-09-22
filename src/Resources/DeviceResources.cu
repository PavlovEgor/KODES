#include "DeviceResources.cuh"

// No KODES_DEFINE_DEVICE_OBJECT here: this class is never a row of a type
// table, only its leaves are. See the head of DeviceResources.cuh.

__host__ void
kodes::DeviceResources::allocate()
{
    // state space: the whole batch, one slot per system
    CUDA_CHECK(cudaMalloc(&vectors, size_t(systemSize_) * ensembleSize_ * sizeof(scalar)));
    CUDA_CHECK(cudaMalloc(&parameters, size_t(parameterSize_) * ensembleSize_ * sizeof(scalar)));

    // scratch space: one slot per resident thread
    CUDA_CHECK(cudaMalloc(&currentVector_, size_t(systemSize_) * scratchSize_ * sizeof(scalar)));
    CUDA_CHECK(cudaMalloc(&currentParameters_, size_t(parameterSize_) * scratchSize_ * sizeof(scalar)));

    StepState::allocate(blocks_);
}

__host__ void
kodes::DeviceResources::deallocate()
{
    CUDA_CHECK(cudaFree(vectors));
    CUDA_CHECK(cudaFree(parameters));

    CUDA_CHECK(cudaFree(currentVector_));
    CUDA_CHECK(cudaFree(currentParameters_));

    StepState::deallocate();
}

__host__ __device__ void
kodes::DeviceResources::printVectori(const label i) const
{
    for (label j = 0; j < systemSize_; ++j) {
        printf("%0.2f ", this->vectors[INDEXSTATE(i, j, ensembleSize_)]);
    }
    printf("\n");
}
