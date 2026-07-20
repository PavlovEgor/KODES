#ifndef basic_types
#define basic_types

typedef double scalar;
typedef int    label;

#define SMALL 1.0e-15
#define GREAT 1.0e+15
#define MAX_VEC_SIZE 128

#define GRID_DIM (blockDim.x * gridDim.x)
#define T_ID (threadIdx.x + blockIdx.x * blockDim.x)
#define INDEXVEC(i) (T_ID + (i) * GRID_DIM)
// #define INDEXMAT(i, j, size) (T_ID + ((i) * (size) + (j)) * GRID_DIM)
#define INDEXMAT(i, j, size) (T_ID + ((i) + (j) * (size)) * GRID_DIM)

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
