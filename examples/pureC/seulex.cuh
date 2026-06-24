#pragma once

#include "basic_linalg.cuh"

typedef struct
{
    scalar**     data;
    label       sizeOfSystem;
    label       numOfSystems;
} ODEVectors;

typedef struct
{
    scalar**    data;
    label       n;
    label       m;
} scalarRectangularMatrix;

typedef struct
{
    scalar**    data;
    label       n;
} scalarSquareMatrix;

typedef struct stepState
{
    bool forward;
    scalar dxTry;
    scalar dxDid;
    bool first;
    bool last;
    bool reject;
    bool prevReject;

    // Конструктор для device
    __device__ __host__
    stepState(const scalar dx)
        : forward(dx > 0.0 ? true : false)
        , dxTry(dx)
        , dxDid(0.0)
        , first(true)
        , last(false)
        , reject(false)
        , prevReject(false)
    {}
} stepState;


__constant__ scalar absTol_    = 1e-5;
__constant__ scalar relTol_    = 1e-5;

__constant__ label sizeOfSystem_    = 8;

__constant__ scalar stepFactor1_ = 0.6,
                    stepFactor2_ = 0.93,
                    stepFactor3_ = 0.1,
                    stepFactor4_ = 4,
                    stepFactor5_ = 0.5,
                    kFactor1_ = 0.7,
                    kFactor2_ = 0.9;

// __constant__ label  kMaxx_ = 12,
//                     iMaxx_ = kMaxx_+1;
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




void init(ODEVectors* vectors);

__device__
void derivatives(const scalar x, const scalar* y, scalar* dydx);

__device__
void jacobian(const scalar x, const scalar* y, scalar* dfdx, scalar* dfdy);




__device__
bool seul (
    const scalar x0,
    const scalar* y0,
    const scalar dxTot,
    const label k,
    scalar* y,
    const scalar* scale,
    scalar** a_,
    scalar** dfdy_
);


__device__ inline
void extrapolate (const label k,const label sizeOfSystem, scalar* table, scalar* y)
{
    for (label j=k-1; j>0; j--)
    {
        for (label i=0; i<sizeOfSystem; i++)
        {
            table[(j-1) * sizeOfSystem + i] =
                table[j * sizeOfSystem + i] + coeff_[k][j]*(table[j * sizeOfSystem + i] - table[(j-1) * sizeOfSystem + i]);
        }
    }

    for (label i=0; i<sizeOfSystem; i++)
    {
        y[i] = table[i] + coeff_[k][0]*(table[i] - y[i]);
    }
}

__global__
void seulex_solve(scalar* data, label numOfSystems, stepState step, scalar xEnd, scalar* resouces_scalar, label* resouces_label);