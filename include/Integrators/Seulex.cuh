



#ifndef Seulex_H
#define Seulex_H

#pragma once

#define kMaxx_ 12
#define iMaxx_ (kMaxx_ + 1)

__constant__ scalar jacRedo_ = 1e-5;

__constant__ label nSeq_[iMaxx_] = {1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128};

__constant__ scalar cpu_[iMaxx_] = {10, 15, 22, 33, 48, 71, 102, 149, 212, 307, 434, 625, 880};

__constant__ scalar coeff_[iMaxx_][iMaxx_] = {
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
            table[INDEXVEC((j-1) * sizeOfSystem + i)] =
                table[INDEXVEC(j * sizeOfSystem + i)] + coeff_[k][j]*(table[INDEXVEC(j * sizeOfSystem + i)] - table[INDEXVEC((j-1) * sizeOfSystem + i)]);
        }
    }

    for (label i=0; i<sizeOfSystem; i++)
    {
        y[INDEXVEC(i)] = table[INDEXVEC(i)] + coeff_[k][0]*(table[INDEXVEC(i)] - y[INDEXVEC(i)]);
    }
}

template<class ODESystem>
__global__
void seulex_solve(ODESystem* ode, kodes::SeulexDeviceResources* res, stepState step);


namespace kodes 
{
template<class ODESystem>
class Seulex
{
    public Integrator<ODESystem, SeulexDeviceResources>;
private:

public:

    Seulex(ODESystem* ode, SeulexDeviceResources* res, stepState step, label numOfSystems);
        
    virtual ~Seulex() = default;

    void solve() override;

};


template<class ODESystem>
Seulex<ODESystem>::Seulex(ODESystem* ode, SeulexDeviceResources* res, stepState step, label numOfSystems)
: Integrator<ODESystem, SeulexDeviceResources>(ode, res, step, numOfSystems) {}

template<class ODESystem>
void Seulex<ODESystem>::solve()
{
    seulex_solve<ODESystem><<<blocks, threads, sharedMemSize>>>(this->ode_, this->res_, this->stepState_);
}

}

#endif
