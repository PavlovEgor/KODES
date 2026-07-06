

#ifndef Integrator_H
#define Integrator_H

#pragma once


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

    stepState step_;

public:

    Integrator(ODESystem* ode, SolverDeviceResources* res, stepState step, label numOfSystems);
        
    virtual ~Integrator() = default;

    virtual void solve() =0;
};


template<class ODESystem, class SolverDeviceResources>
Integrator<ODESystem, SolverDeviceResources>::Integrator(ODESystem* ode, SolverDeviceResources* res, stepState step, label numOfSystems)
: ode_(ode), res_(res), step_(step)
{
    threads = numOfSystems <= 256 ? numOfSystems : 256;
    blocks = cuda::ceil_div(host_res.numOfSystems(), threads);
    sharedMemSize = (3 * threads + threads) * sizeof(scalar); 
}

}

#endif
