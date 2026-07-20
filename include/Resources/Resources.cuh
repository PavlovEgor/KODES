#ifndef KODES_RESOURCES_CUH
#define KODES_RESOURCES_CUH

#include "basic_types.cuh"
#include <stdio.h>

namespace kodes 
{
class Resources
{
protected:
    label numOfSystems_;
    label sizeOfSystem_;
    label numOfParameters_;
    // Stride between components in the flat device buffers: the smallest
    // multiple of the kernel block size that is >= numOfSystems_, i.e. the
    // GRID_DIM every kernel touching those buffers is launched with. Kept
    // distinct from numOfSystems_ (the real/logical system count used for
    // bounds checks and host transfers) so a numOfSystems_ that isn't a
    // multiple of the block size doesn't misalign INDEXVEC/INDEXMAT.
    label gridSize_;

public:
    __device__ __host__
    Resources(const label numOfSystems, const label sizeOfSystem, const label numOfParameters)
        : numOfSystems_(numOfSystems), sizeOfSystem_(sizeOfSystem), numOfParameters_(numOfParameters)
        , gridSize_(kodes::paddedNumOfSystems(numOfSystems)) {}

    __device__ __host__
    virtual ~Resources() = default;

    __device__ __host__ label numOfSystems() { return numOfSystems_; }
    __device__ __host__ label sizeOfSystem() { return sizeOfSystem_; }
    __device__ __host__ label numOfParameters() { return numOfParameters_; }
    __device__ __host__ label gridSize() { return gridSize_; }
};
}
#endif
