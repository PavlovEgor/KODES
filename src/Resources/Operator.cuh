#ifndef KODES_OPERATOR
#define KODES_OPERATOR

#pragma once

#include "basicTypes.cuh"
#include "HostResources.cuh"
#include "DeviceResources.cuh"

namespace kodes
{

// Moves one batch of state between the host's per-component pointers and the
// device's flat state space, in both directions.
//
// It only ever touches vectors/parameters and the three sizes, all of which
// belong to the base classes, so it no longer cares which DeviceResources
// subclass the chosen method brought with it.
class Operator
{
protected:

    HostResources*       hostRes_;
    DeviceResources*     deviceRes_;

    label                    ensembleSize_;
    label                    systemSize_;
    label                    parameterSize_;

    label                    batchSize_;
    label                    lastBatchSize_;
    label                    lastBatchIndex_;

public:

    Operator(HostResources* hostRes, DeviceResources* deviceRes);

    virtual ~Operator() = default;

    virtual void cpyHostToDevice(label batchIndex);
    virtual void cpyDeviceToHost(label batchIndex);

    // Systems in that batch - the last one is normally shorter than batchSize
    virtual label getRealBatchSize(label batchIndex);
};

}

#endif
