#include "basic_linalg.cuh"


__device__
void LUDecompose (scalar* matrix, label* pivotIndices, const label size)
{
    int sign;
    LUDecompose(matrix, pivotIndices, size, &sign);
}
