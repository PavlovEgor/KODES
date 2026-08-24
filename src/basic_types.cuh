#ifndef basic_types
#define basic_types

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef double scalar;
typedef int    label;

#define SMALL 1.0e-15
#define GREAT 1.0e+15
#define MAX_VEC_SIZE 256

// Two distinct address spaces are used on the device:
//
//  * scratch space - the per thread temporaries (Jacobian, LU matrix, the
//    extrapolation table, ...). One slot per *resident* thread, stride
//    GRID_DIM == the number of threads actually launched. Addressed with
//    INDEXVEC/INDEXMAT below, which is also the layout pyJac's generated code
//    expects (its INDEX macro is identical).
//
//  * state space - the per system storage of one batch (vectors, parameters
//    and the step state). One slot per system, stride batchSize, which is
//    normally much larger than the number of resident threads. Addressed with
//    INDEXSTATE.
//
// A thread walks its share of the batch in a grid-stride loop, moving one
// system at a time between the two spaces (DeviceResources::loadSystem /
// storeSystem).

#define GRID_DIM (blockDim.x * gridDim.x)
#define T_ID (threadIdx.x + blockIdx.x * blockDim.x)

// scratch space: slot of the calling thread, component i
#define INDEXVEC(i) (T_ID + (i) * GRID_DIM)
#define INDEXMAT(i, j, size) (T_ID + ((i) + (j) * (size)) * GRID_DIM)

// state space: system `sys` of the batch, component i
#define INDEXSTATE(sys, i, batchSize) ((sys) + (i) * (batchSize))

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
