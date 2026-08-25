#ifndef KODES_BASIC_TYPES
#define KODES_BASIC_TYPES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef double scalar;
typedef int    label;

#define SMALL 1.0e-15
#define GREAT 1.0e+15
#define MAX_VEC_SIZE 256

#define GRID_DIM (blockDim.x * gridDim.x)
#define T_ID (threadIdx.x + blockIdx.x * blockDim.x)

// scratch space: the calling thread's slot, one per resident thread
#define INDEXVEC(i) (T_ID + (i) * GRID_DIM)
#define INDEXMAT(i, j, size) (T_ID + ((i) + (j) * (size)) * GRID_DIM)

// state space: system `system` of the batch, one slot per system
#define INDEXSTATE(system, i, batchSize) ((system) + (i) * (batchSize))

#define KODES_BLOCK_SIZE 256

#define CUDA_CHECK(call)                                                          \
    do {                                                                         \
        cudaError_t err__ = (call);                                              \
        if (err__ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(err__));                                  \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                        \
    } while (0)

#define CUDA_CHECK_LAST()                                                         \
    do {                                                                         \
        cudaError_t err__ = cudaGetLastError();                                  \
        if (err__ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA kernel launch error at %s:%d: %s\n", __FILE__, \
                    __LINE__, cudaGetErrorString(err__));                        \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                        \
    } while (0)

namespace kodes
{
    __host__ __device__ inline label blockSize(const label ensembleSize)
    {
        return ensembleSize < KODES_BLOCK_SIZE ? ensembleSize : KODES_BLOCK_SIZE;
    }

    __host__ __device__ inline label numOfBlocks(const label ensembleSize)
    {
        const label threads = blockSize(ensembleSize);
        return (ensembleSize + threads - 1) / threads;
    }

    // Dynamic shared memory of one block: the 4*blockDim.x scratch doubles
    // pyJac's generated kernels expect (shared_temp)
    __host__ __device__ inline size_t sharedMemorySize(const label threads)
    {
        return size_t(4 * threads) * sizeof(scalar);
    }
}


#endif
