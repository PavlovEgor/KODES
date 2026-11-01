#ifndef KODES_RODAS34
#define KODES_RODAS34
#pragma once

#include "Rosenbrock.cuh"

namespace kodes
{

// OpenFOAM's rodas34: order 4, error estimate of order 3, 6 stages, L-stable and
// stiffly accurate.
// The whole method is Rosenbrock; this names its tableau.
//
// Selected by the name "rodas34", see methodTable.
class Rodas34
    :
    public Rosenbrock<Rodas34Tableau>
{
public:

    __device__ __host__
    Rodas34
    (
        const label batchSize,
        const label scratchSize,
        const label blocks,
        const label systemSize,
        const label parameterSize
    )
        : Rosenbrock(batchSize, scratchSize, blocks, systemSize, parameterSize)
    {}

    KODES_DECLARE_DEVICE_OBJECT(Rodas34)
};

}

#endif
