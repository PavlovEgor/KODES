#include "primitiveSolverForTests.cuh"

cudaError_t createSolver(SolverHandle** solver, cudaStream_t stream) {
    cudaError_t err = cudaSuccess;
    
    // Выделяем память для handle
    *solver = (mulTwoSolver*)malloc(sizeof(mulTwoSolver));
    if (*solver == nullptr) {
        return cudaErrorMemoryAllocation;
    }
    
    // Создаем поток CUDA
    err = cudaStreamCreate(stream);
    if (err != cudaSuccess) {
        free(*solver);
        return err;
    }
    
    // Выделяем память для структуры на GPU
    err = cudaMalloc(&(*solver)->d_state, sizeof(BasicChemistryState));
    if (err != cudaSuccess) {
        cudaStreamDestroy(stream);
        free(*solver);
        return err;
    }
    
    // Выделяем память для массива концентраций на GPU
    err = cudaMalloc(&(*solver)->d_concentrations, numberOfComponents * sizeof(double));
    if (err != cudaSuccess) {
        cudaFree((*solver)->d_state);
        cudaStreamDestroy((*solver)->stream);
        free(*solver);
        return err;
    }
    
    // Инициализируем структуру на GPU
    BasicChemistryState temp_state;
    temp_state.numberOfComponents = numberOfComponents;
    temp_state.concentrations = (*solver)->d_concentrations;
    
    err = cudaMemcpyAsync(
        (*solver)->d_state,
        &temp_state,
        sizeof(BasicChemistryState),
        cudaMemcpyHostToDevice,
        (*solver)->stream
    );
    
    if (err != cudaSuccess) {
        cudaFree((*solver)->d_concentrations);
        cudaFree((*solver)->d_state);
        cudaStreamDestroy((*solver)->stream);
        free(*solver);
        return err;
    }
    
    (*solver)->numberOfComponents = numberOfComponents;
    
    return err;
}