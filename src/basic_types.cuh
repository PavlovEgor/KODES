#ifndef basic_types
#define basic_types

typedef double scalar;
typedef int    label;

#define SMALL 1.0e-15
#define GREAT 1.0e+15
#define MAX_VEC_SIZE 256

#define GRID_DIM (blockDim.x * gridDim.x)
#define T_ID (threadIdx.x + blockIdx.x * blockDim.x)
#define INDEXVEC(i) (T_ID + (i) * GRID_DIM)
#define INDEXMAT(i, j, size) (T_ID + ((i) + (j) * (size)) * GRID_DIM)

#define KODES_BLOCK_SIZE 256

namespace kodes
{
    __host__ __device__ inline label blockSize(const label numOfSystems)
    {
        return numOfSystems < KODES_BLOCK_SIZE ? numOfSystems : KODES_BLOCK_SIZE;
    }

    __host__ __device__ inline label numOfBlocks(const label numOfSystems)
    {
        const label threads = blockSize(numOfSystems);
        return (numOfSystems + threads - 1) / threads;
    }

    __host__ __device__ inline label sharedMemorySize(const label numOfSystems)
    {
        const label threads = blockSize(numOfSystems);
        return (3 * threads + threads) * sizeof(scalar);
    }
}


#endif
