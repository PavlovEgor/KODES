#ifndef KODES_ROSENBROCK34
#define KODES_ROSENBROCK34
#pragma once

#include "Rosenbrock.cuh"

namespace kodes
{

// OpenFOAM's Rosenbrock34: order 4, error estimate of order 3, 4 stages, L-stable.
// The whole method is Rosenbrock; this names its tableau.
//
// Selected by the name "rosenbrock34", see methodTable.
class Rosenbrock34
    :
    public Rosenbrock<Rosenbrock34Tableau>
{
public:

    __device__ __host__
    Rosenbrock34
    (
        const label batchSize,
        const label scratchSize,
        const label blocks,
        const label systemSize,
        const label parameterSize
    )
        : Rosenbrock(batchSize, scratchSize, blocks, systemSize, parameterSize)
    {}

    KODES_DECLARE_DEVICE_OBJECT(Rosenbrock34)
};

}

#endif
