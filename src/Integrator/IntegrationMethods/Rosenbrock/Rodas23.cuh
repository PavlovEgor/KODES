#ifndef KODES_RODAS23
#define KODES_RODAS23
#pragma once

#include "Rosenbrock.cuh"

namespace kodes
{

// OpenFOAM's rodas23: order 3, error estimate of order 2, 4 stages, L-stable and
// stiffly accurate.
// The whole method is Rosenbrock; this names its tableau.
//
// Selected by the name "rodas23", see methodTable.
class Rodas23
    :
    public Rosenbrock<Rodas23Tableau>
{
public:

    __device__ __host__
    Rodas23
    (
        const label batchSize,
        const label scratchSize,
        const label blocks,
        const label systemSize,
        const label parameterSize
    )
        : Rosenbrock(batchSize, scratchSize, blocks, systemSize, parameterSize)
    {}

    KODES_DECLARE_DEVICE_OBJECT(Rodas23)
};

}

#endif
