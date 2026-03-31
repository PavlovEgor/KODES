#ifndef SOLVER_H
#define SOLVER_H

#include <cuda_runtime.h>
#include "ChemistryState.cuh"

typedef struct {
    double coef;
} mulSolver;

cudaError_t createSolver(mulTwoSolver** solver, BasicChemistryState* d_state, cudaStream_t stream, double coef);

cudaError_t destroySolver(mulTwoSolver* solver);

cudaError_t copyToSolver(mulTwoSolver* solver, const BasicChemistryState* h_src);

cudaError_t copyFromSolver(BasicChemistryState* h_dest, const mulTwoSolver* solver);

cudaError_t solve(mulTwoSolver* solver);


#endif