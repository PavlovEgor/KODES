



#ifndef Seulex_H
#define Seulex_H

#include <cuda/cmath>
#include <cuda_runtime.h>

#include "basic_linalg.cuh"
// #include "SeulexDeviceResources.cuh"
#include "Integrator.cuh"

#pragma once

__constant__ static scalar absTol_    = 1e-5;
__constant__ static scalar relTol_    = 1e-1;

__constant__ static scalar stepFactor1_ = 0.6,
                    stepFactor2_ = 0.93,
                    stepFactor3_ = 0.1,
                    stepFactor4_ = 4,
                    stepFactor5_ = 0.5,
                    kFactor1_ = 0.7,
                    kFactor2_ = 0.9;

#define kMaxx_ 12
#define iMaxx_ (kMaxx_ + 1)

__constant__ static scalar jacRedo_ = 1e-5;

__constant__ static label nSeq_[iMaxx_] = {1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128};

__constant__ static scalar cpu_[iMaxx_] = {10, 15, 22, 33, 48, 71, 102, 149, 212, 307, 434, 625, 880};

__constant__ static scalar coeff_[iMaxx_][iMaxx_] = {
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {2.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {1.0, 3.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0.5, 1.0, 2.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0.3333333333333333, 0.6, 1.0, 3.0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0.2, 0.3333333333333333, 0.5, 1.0, 2.0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0.14285714285714285, 0.23076923076923078, 0.3333333333333333, 0.6, 1.0, 3.0, 0, 0, 0, 0, 0, 0, 0},
    {0.09090909090909091, 0.14285714285714285, 0.2, 0.3333333333333333, 0.5, 1.0, 2.0, 0, 0, 0, 0, 0, 0},
    {0.06666666666666667, 0.10344827586206898, 0.14285714285714285, 0.23076923076923078, 0.3333333333333333, 0.6, 1.0, 3.0, 0, 0, 0, 0, 0},
    {0.043478260869565216, 0.06666666666666667, 0.09090909090909091, 0.14285714285714285, 0.2, 0.3333333333333333, 0.5, 1.0, 2.0, 0, 0, 0, 0},
    {0.03225806451612903, 0.049180327868852465, 0.06666666666666667, 0.10344827586206898, 0.14285714285714285, 0.23076923076923078, 0.3333333333333333, 0.6, 1.0, 3.0, 0, 0, 0},
    {0.02127659574468085, 0.03225806451612903, 0.043478260869565216, 0.06666666666666667, 0.09090909090909091, 0.14285714285714285, 0.2, 0.3333333333333333, 0.5, 1.0, 2.0, 0, 0},
    {0.015873015873015872, 0.024, 0.03225806451612903, 0.049180327868852465, 0.06666666666666667, 0.10344827586206898, 0.14285714285714285, 0.23076923076923078, 0.3333333333333333, 0.6000000000000001, 1.0, 3.000000000000001, 0}
};

template<class ODESystem>
__device__
bool seul (
    kodes::SeulexDeviceResources* res,
    ODESystem* ode,
    const scalar x0,
    const scalar dxTot,
    const label k,
    scalar theta
);


__device__ inline
void extrapolate (const label k,const label sizeOfSystem, scalar* table, scalar* y)
{
    for (label j=k-1; j>0; j--)
    {
        for (label i=0; i<sizeOfSystem; i++)
        {
            table[INDEXMAT(i, j-1, sizeOfSystem)] =
                table[INDEXMAT(i, j, sizeOfSystem)] + coeff_[k][j]*(table[INDEXMAT(i, j, sizeOfSystem)] - table[INDEXMAT(i, j-1, sizeOfSystem)]);
        }
    }

    for (label i=0; i<sizeOfSystem; i++)
    {
        y[INDEXVEC(i)] = table[INDEXMAT(i, 0, sizeOfSystem)] + coeff_[k][0]*(table[INDEXMAT(i, 0, sizeOfSystem)] - y[INDEXVEC(i)]);
    }
}

template<class ODESystem>
__global__
void seulex_solve(ODESystem* ode, kodes::SeulexDeviceResources* res, stepState step);


namespace kodes 
{
template<class ODESystem>
class Seulex
: public Integrator<ODESystem, SeulexDeviceResources>
{
    
private:

public:

    Seulex(ODESystem* ode, SeulexDeviceResources* res, label numOfSystems);
        
    virtual ~Seulex() = default;

    void solve(stepState step) override;

};

}

#include "Seulex.cu"

#endif
