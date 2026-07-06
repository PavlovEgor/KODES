#ifndef basic_types
#define basic_types

typedef double scalar;
typedef int    label;

#define GRID_DIM (blockDim.x * gridDim.x)
#define T_ID (threadIdx.x + blockIdx.x * blockDim.x)
#define INDEXVEC(i) (T_ID + (i) * GRID_DIM)
#define INDEXMAT(i, j, size) (T_ID + ((i) * (size) + (j)) * GRID_DIM)

#endif