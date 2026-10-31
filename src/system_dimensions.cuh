#ifndef KODES_SYSTEM_DIMENSIONS
#define KODES_SYSTEM_DIMENSIONS

#pragma once

#include "basic_types.cuh"

// Generated for the mechanism of the build by kodes_pyjac_mechanism() in
// cmake/kodes.cmake
#include "mechanism_dimensions.h"

namespace kodes
{

// The size of a system and of its parameters. A library is built for one
// mechanism (see kodes_library in cmake/kodes.cmake), so both are known when it
// is compiled, and every view takes its size from its type: a loop over a
// vector has a constant trip count, and no view carries a size of its own.
//
// Which is why the loops over a system in the views and in Seulex say
// `#pragma unroll 4`. Left to itself, nvcc unrolls a loop of a known short
// trip count completely - h2o2's 9 - and in Seulex::step that hoists so many
// loads that the registers run out: its stack went from 352 to 808 bytes, all
// of it spills. Unrolled by four, every function of both mechanisms is at or
// below the stack it had while the size was a run time value.
constexpr label kSystemSize = KODES_SYSTEM_SIZE;
constexpr label kParameterSize = KODES_PARAMETER_SIZE;

// The sizes still arrive at run time too, from whoever fills the batch. Where
// they enter the library they are checked against the ones it was built for.
__host__ inline void checkSystemDimensions
(
    const label systemSize,
    const label parameterSize,
    const char* caller
)
{
    if (systemSize != kSystemSize || parameterSize != kParameterSize)
    {
        fprintf
        (
            stderr,
            "%s error: systems of %d components and %d parameters, but the "
            "library is built for %d and %d\n",
            caller, systemSize, parameterSize, kSystemSize, kParameterSize
        );
        std::exit(EXIT_FAILURE);
    }
}

}

#endif
