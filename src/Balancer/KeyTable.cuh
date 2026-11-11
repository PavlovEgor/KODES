#ifndef KODES_KEY_TABLE
#define KODES_KEY_TABLE

#pragma once

#include "basic_types.cuh"
#include "DeviceBuffer.cuh"

namespace kodes
{

// The keys a balancer orders a batch by, numOfKeys of them per system: key k of
// a system is at INDEXSTATE(system, k, batchSize), the layout of BatchArray.
// Not a BatchArray, because there the number of components is fixed when the
// library is built, and here it is the balancer's, which a run chooses. A view,
// see DeviceBuffer.cuh: it owns nothing and cannot be assigned.
class KeyTable
    :
    public DeviceBuffer<scalar>
{
    label batchSize_;

public:

    KeyTable() = default;

    KeyTable(const KeyTable&) = default;

    KeyTable& operator=(const KeyTable&) = delete;

    __device__ scalar& operator()(const label system, const label k) const
    {
        return this->data_[INDEXSTATE(system, k, batchSize_)];
    }

    __host__ void allocate(const label numOfKeys, const label systems)
    {
        batchSize_ = systems;
        this->allocateElements(size_t(numOfKeys) * systems);
    }
};

}

#endif
