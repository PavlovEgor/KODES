#ifndef KODES_ROSENBROCK23
#define KODES_ROSENBROCK23
#pragma once

#include "Rosenbrock.cuh"

namespace kodes
{

// OpenFOAM's Rosenbrock23: order 3, error estimate of order 2, 3 stages, L-stable.
// The whole method is Rosenbrock; this names its tableau.
//
// Selected by the name "rosenbrock23", see methodTable.
class Rosenbrock23
    :
    public Rosenbrock<Rosenbrock23Tableau>
{
public:

    __device__ __host__
    Rosenbrock23
    (
        const label batchSize,
        const label scratchSize,
        const label blocks,
        const label systemSize,
        const label parameterSize
    )
        : Rosenbrock(batchSize, scratchSize, blocks, systemSize, parameterSize)
    {}

    KODES_DECLARE_DEVICE_OBJECT(Rosenbrock23)
};

}

#endif
