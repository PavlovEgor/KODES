

#ifndef Integrator_H
#define Integrator_H

#pragma once

#include "basic_types.cuh"


namespace kodes 
{
template<class ODESystem, class SolverDeviceResources>
class Integrator
{

protected:
    label threads;
    label blocks;
    size_t sharedMemSize;

    ODESystem* ode_;
    SolverDeviceResources* resources_;

public:

    Integrator(ODESystem* ode, SolverDeviceResources* resources, label batchSize);
        
    virtual ~Integrator() = default;

    virtual void solve(kodes::stepState step, label realBatchSize) =0;
};


template<class ODESystem, class SolverDeviceResources>
Integrator<ODESystem, SolverDeviceResources>::Integrator(ODESystem* ode, SolverDeviceResources* resources, label batchSize)
: ode_(ode), resources_(resources)
{
    threads = kodes::blockSize(batchSize);
    blocks = kodes::numOfBlocks(batchSize);
    sharedMemSize = kodes::sharedMemorySize(batchSize);

    if (batchSize != threads * blocks)
    {
        printf("batchSize != threads * blocks");
    }
}

}

#endif
