

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
    SolverDeviceResources* res_;

public:

    Integrator(ODESystem* ode, SolverDeviceResources* res, label numOfSystems);
        
    virtual ~Integrator() = default;

    virtual void solve(stepState step) =0;
};


template<class ODESystem, class SolverDeviceResources>
Integrator<ODESystem, SolverDeviceResources>::Integrator(ODESystem* ode, SolverDeviceResources* res, label numOfSystems)
: ode_(ode), res_(res)
{
    threads = kodes::blockSize(numOfSystems);
    blocks = kodes::numOfBlocks(numOfSystems);
    sharedMemSize = kodes::sharedMemorySize(threads);
}

}

#endif
