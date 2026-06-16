#pragma once

typedef double scalar;
typedef int    label;

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
void derivatives(scalar x, scalar* y, scalar* dydx);

__device__
void jacobian(scalar x, scalar* y, scalar* dfdx, scalar* dfdy);

__device__
scalar solve(scalar x0, scalar* y0, scalar* dydx0, scalar dx, scalar* y);

__device__
scalar normalizeError (scalar* y0, scalar* y, scalar* err);

__device__
scalar clamp (scalar scale, scalar minScale, scalar maxScale);

__global__
void euler_solve(scalar* data, label numOfSystems, scalar xStart, scalar xEnd);