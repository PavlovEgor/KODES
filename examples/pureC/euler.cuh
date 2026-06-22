#pragma once

#include "basic_linalg.cuh"


__constant__ scalar safeScale_ = 0.9;
__constant__ scalar alphaInc_  = 0.2;
__constant__ scalar alphaDec_  = 0.25;
__constant__ scalar minScale_  = 0.2;
__constant__ scalar maxScale_  = 10;
__constant__ scalar absTol_    = 1e-5;
__constant__ scalar relTol_    = 1e-5;

__constant__ label sizeOfSystem    = 8;

typedef struct
{
    scalar*     data;
    label       sizeOfSystem;
    label       numOfSystems;
} ODEVectors;


void init(ODEVectors* vectors);

__device__
void derivatives(const scalar x, const scalar* y, scalar* dydx);

__device__
void jacobian(const scalar x, const scalar* y, scalar* dfdx, scalar* dfdy);

__device__
scalar solve(const scalar x0, const scalar* y0, const scalar* dydx0, scalar dx, scalar* y);

__global__
void euler_solve(scalar* data, const label numOfSystems, const scalar xStart, const scalar xEnd, scalar* resouces);