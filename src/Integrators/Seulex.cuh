



#ifndef Seulex_H
#define Seulex_H

#include <cuda/cmath>
#include <cuda_runtime.h>

#include "basic_linalg.cuh"
#include "Integrator.cuh"

#pragma once

__constant__ static scalar stepFactor1_ = 0.6,
                    stepFactor2_ = 0.93,
                    stepFactor3_ = 0.1,
                    stepFactor4_ = 4,
                    stepFactor5_ = 0.5,
                    kFactor1_ = 0.7,
                    kFactor2_ = 0.9;

#define kMaxx_ 12
#define iMaxx_ (kMaxx_ + 1)

// nSeq_[0] = 2, nSeq_[1] = 3, nSeq_[i] = 2*nSeq_[i-2], as built by the OpenFOAM
// constructor. cpu_ and coeff_ below are derived from this sequence
__constant__ static label nSeq_[iMaxx_] = {2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128};

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
// Per system cycle counters for the cost centres of the algorithm. The four
// timed parts are exactly the ones the cpu_ weights above model, with the
// nominal OpenFOAM ratios cpuJac : cpuFunc : cpuLU : cpuSolve = 5 : 1 : 1 : 1
struct SeulexProfile
{
    long long total;
    long long jacobian;
    long long derivatives;
    long long luDecompose;
    long long luBacksubstitute;
    long long linSolve;

    label nJacobian;
    label nDerivatives;
    label nLuDecompose;
    label nLuBacksubstitute;
    label nLinSolve;
    label nLinIters;
    label nLinFail;
    label nSeul;
    label nStep;
    label nReject;

    __device__
    SeulexProfile()
    :
        total(0), jacobian(0), derivatives(0), luDecompose(0), luBacksubstitute(0),
        linSolve(0),
        nJacobian(0), nDerivatives(0), nLuDecompose(0), nLuBacksubstitute(0),
        nLinSolve(0), nLinIters(0), nLinFail(0),
        nSeul(0), nStep(0), nReject(0)
    {}

    __device__
    void print(const label system) const
    {
        const scalar pct = total > 0 ? 100.0/total : 0.0;

        const long long other =
            total - jacobian - derivatives - luDecompose - luBacksubstitute
          - linSolve;

        printf
        (
            "\n"
            "seulex profile, system %d \n"
            "                        cycles      share      calls   cycles/call \n"
            "  jacobian        %12lld  %8.2f%%  %9d  %12lld \n"
            "  derivatives     %12lld  %8.2f%%  %9d  %12lld \n"
            "  LU decompose    %12lld  %8.2f%%  %9d  %12lld \n"
            "  LU backsubst    %12lld  %8.2f%%  %9d  %12lld \n"
            "  Bi-CGStab       %12lld  %8.2f%%  %9d  %12lld \n"
            "  other           %12lld  %8.2f%% \n"
            "  total           %12lld \n"
            "  steps %d, rejected %d, seul() calls %d \n"
            "  Bi-CGStab iterations %d (%.2f per solve), fallbacks %d \n"
            "\n",
            system,
            jacobian, jacobian*pct, nJacobian,
                nJacobian ? jacobian/nJacobian : 0LL,
            derivatives, derivatives*pct, nDerivatives,
                nDerivatives ? derivatives/nDerivatives : 0LL,
            luDecompose, luDecompose*pct, nLuDecompose,
                nLuDecompose ? luDecompose/nLuDecompose : 0LL,
            luBacksubstitute, luBacksubstitute*pct, nLuBacksubstitute,
                nLuBacksubstitute ? luBacksubstitute/nLuBacksubstitute : 0LL,
            linSolve, linSolve*pct, nLinSolve,
                nLinSolve ? linSolve/nLinSolve : 0LL,
            other, other*pct,
            total,
            nStep, nReject, nSeul,
            nLinIters, nLinSolve ? scalar(nLinIters)/nLinSolve : 0.0, nLinFail
        );
    }
};

template<class ODESystem>
__device__
bool seul (
    kodes::SeulexDeviceResources* resources,
    ODESystem* ode,
    const scalar t0,
    const scalar dtTot,
    const label k,
    scalar& theta,
    const kodes::IntegratorControls& controls,
    SeulexProfile& profile
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
void seulex_solve
(
    ODESystem* ode,
    kodes::SeulexDeviceResources* resources,
    scalar deltaT,
    label realBatchSize,
    kodes::IntegratorControls controls,
    label profileSystem,
    bool    firstBatch
);


namespace kodes
{
template<class ODESystem>
class Seulex
: public Integrator<ODESystem, SeulexDeviceResources>
{

private:
    // Index within the batch whose cycle breakdown is printed at the end of the
    // kernel, negative to keep the kernel quiet
    label profileSystem_;
public:

    Seulex
    (
        ODESystem* ode,
        SeulexDeviceResources* resources,
        label batchSize,
        const IntegratorControls& controls = IntegratorControls()
    );

    virtual ~Seulex() = default;

    void solve(scalar deltaT, label realBatchSize, bool firstBatch) override;

    void setProfileSystem(const label system) { profileSystem_ = system; }

};

}

#include "Seulex.cu"

#endif
