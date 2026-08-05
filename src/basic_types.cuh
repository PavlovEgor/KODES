#ifndef basic_types
#define basic_types

#include <cstdio>
#include <cstdlib>

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

    __host__ __device__ inline label sharedMemorySize(const label ensembleSize)
    {
        const label threads = blockSize(ensembleSize);
        return (3 * threads + threads) * sizeof(scalar);
    }
}


#endif
