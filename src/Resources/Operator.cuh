#ifndef KODES_OPERATOR
#define KODES_OPERATOR

#pragma once

#include "basic_types.cuh"
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

    // what setNumOfSystems() was last told, and what follows from it
    label                    numOfSystems_;
    label                    lastBatchSize_;
    label                    lastBatchIndex_;

public:

    // Starts out with the whole ensemble of hostRes
    Operator(HostResources* hostRes, DeviceResources* deviceRes);

    virtual ~Operator() = default;

    // Transfer only the first numOfSystems of the ensemble, 0 to all of it:
    // a caller that leaves some systems out - those it knows will not react -
    // packs the rest at the front. Every batch but the last is full, and with
    // no systems there is no batch.
    void setNumOfSystems(label numOfSystems);

    label numOfSystems() const { return numOfSystems_; }

    label numOfBatches() const { return lastBatchIndex_ + 1; }

    virtual void cpyHostToDevice(label batchIndex);
    virtual void cpyDeviceToHost(label batchIndex);

    // Systems in that batch - the last one is normally shorter than batchSize
    virtual label getRealBatchSize(label batchIndex);
};

}

#endif
