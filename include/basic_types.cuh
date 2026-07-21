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
// #define INDEXMAT(i, j, size) (T_ID + ((i) * (size) + (j)) * GRID_DIM)
#define INDEXMAT(i, j, size) (T_ID + ((i) + (j) * (size)) * GRID_DIM)

// GRID_DIM above is *not* numOfSystems: it is blockDim.x*gridDim.x as launched.
// INDEXVEC/INDEXMAT use GRID_DIM as the stride between components, so every
// buffer addressed through them must be allocated with that same stride, and
// every kernel that touches them must be launched so blockDim.x*gridDim.x
// comes out to exactly that stride - otherwise systems overlap in memory or
// go out of bounds whenever numOfSystems isn't a multiple of the block size.
// These helpers are the single place that stride ("grid size") is derived
// from numOfSystems, so allocation (Resources) and launch configuration
// (Integrator) can't disagree.
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

    // Smallest multiple of the launch's block size that is >= numOfSystems.
    // Always equal to blockDim.x*gridDim.x for a kernel launched with
    // (numOfBlocks(numOfSystems), blockSize(numOfSystems)), i.e. to GRID_DIM
    // as seen from inside that kernel.
    __host__ __device__ inline label paddedNumOfSystems(const label numOfSystems)
    {
        return blockSize(numOfSystems) * numOfBlocks(numOfSystems);
    }
}

typedef struct stepState
{
    bool forward;
    scalar dxTry;
    scalar dxDid;
    bool first;
    bool last;
    bool reject;
    bool prevReject;

    __device__ __host__
    stepState(const scalar dx)
        : forward(dx > 0.0 ? true : false)
        , dxTry(dx)
        , dxDid(0.0)
        , first(true)
        , last(false)
        , reject(false)
        , prevReject(false)
    {}
} stepState;

#endif
