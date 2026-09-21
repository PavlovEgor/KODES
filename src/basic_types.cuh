#ifndef KODES_BASIC_TYPES
#define KODES_BASIC_TYPES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef double scalar;
typedef int    label;

#define SMALL 1.0e-15
#define GREAT 1.0e+15

#define GRID_DIM (blockDim.x * gridDim.x)
#define T_ID (threadIdx.x + blockIdx.x * blockDim.x)

#define GRID_BLOCKS (gridDim.x)
#define B_ID (blockIdx.x)

// scratch space: the calling thread's slot, one per resident thread
#define INDEXVEC(i) (T_ID + (i) * GRID_DIM)
#define INDEXMAT(i, j, size) (T_ID + ((i) + (j) * (size)) * GRID_DIM)

// block scratch space: the calling block's slot, one per block launched
#define INDEXBLOCKVEC(i) (B_ID + (i) * GRID_BLOCKS)

// A matrix shared by the whole block. Flat on purpose: the base pointer is
// already the block's own, whether it came from shared memory or from the
// block's slot of a global buffer, and the method that uses it cannot tell
// which. It also makes the reads of a collective factorization contiguous.
#define INDEXBLOCKMAT(i, j, size) ((i) + (j) * (size))

// The Jacobian as pyJac wrote it, in the scratch slot of thread `tid`. One
// thread of the block fills it and the whole block reads it back from there,
// so the slot is not the caller's - see Seulex::step.
#define INDEXJACMAT(i, j, size, tid) ((tid) + ((i) + (j) * (size)) * GRID_DIM)

// state space: system `system` of the batch, one slot per system
#define INDEXSTATE(system, i, batchSize) ((system) + (i) * (batchSize))

#define KODES_BLOCK_SIZE 256

// Largest block the library will be asked to launch. Only sizes the staging
// array of a block reduction, so raising it costs a few bytes of shared memory.
#define KODES_MAX_BLOCK_SIZE 1024

// Lanes of a warp. Fixed on every NVIDIA device, and needed as a constant so
// that the shuffle reductions of a balancing pass and of a block reduction can
// be unrolled.
#define KODES_WARP 32

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
    // pyJac's generated kernels expect (shared_temp), and behind them the
    // block's own scratch when the plan decided it fits there. pyJac's half
    // comes first, since its generated code indexes shared_temp from zero.
    __host__ __device__ inline size_t
    sharedMemorySize(const label threads, const size_t blockScratchBytes = 0)
    {
        return size_t(4 * threads) * sizeof(scalar) + blockScratchBytes;
    }

    // Where the block's own scratch starts inside that array
    __host__ __device__ inline size_t blockScratchOffset(const label threads)
    {
        return size_t(4 * threads);
    }
}


#endif
