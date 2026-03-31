#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)


typedef struct {
    int numberOfComponents;
    double* concentrations;
} BasicChemistryState;


typedef struct {
    double coef;
} mulTwoSolver;


__global__ 
void SolveReactorsSystem(mulTwoSolver solver, BasicChemistryState* cells, int numberOfCells) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < numberOfCells) {
        BasicChemistryState* cell = &cells[idx];
        
        for (int j = 0; j < cell->numberOfComponents; j++) {
            cell->concentrations[j] = solver.coef * cell->concentrations[j];
        }
    }
}

int main() {
    int numberOfCells = 1 << 20;
    int numberOfComponents = 10;

    mulTwoSolver solver;
    double coef = 2.0;
    solver.coef = coef;

    BasicChemistryState* h_cellsPull = (BasicChemistryState*)malloc(numberOfCells * sizeof(BasicChemistryState));
    BasicChemistryState* h_cellsPull_res = (BasicChemistryState*)malloc(numberOfCells * sizeof(BasicChemistryState));
    BasicChemistryState* d_cellsPull;
    
    if (!h_cellsPull || !h_cellsPull_res) {
        std::cerr << "Failed to allocate host memory" << std::endl;
        return 1;
    }
    
    double** h_original_concentrations = (double**)malloc(numberOfCells * sizeof(double*));
    if (!h_original_concentrations) {
        std::cerr << "Failed to allocate host pointer array" << std::endl;
        return 1;
    }

    for (int i = 0; i < numberOfCells; i++) {
        h_cellsPull[i].numberOfComponents = numberOfComponents;
        h_cellsPull[i].concentrations = (double*)malloc(numberOfComponents * sizeof(double));
        
        if (!h_cellsPull[i].concentrations) {
            std::cerr << "Failed to allocate concentrations for cell " << i << std::endl;
            return 1;
        }
        
        h_original_concentrations[i] = h_cellsPull[i].concentrations;
        
        for (int j = 0; j < numberOfComponents; j++) {
            h_cellsPull[i].concentrations[j] = (double)i;
        }
    }

    CUDA_CHECK(cudaMalloc(&d_cellsPull, numberOfCells * sizeof(BasicChemistryState)));
    
    double** d_concentrations = (double**)malloc(numberOfCells * sizeof(double*));
    if (!d_concentrations) {
        std::cerr << "Failed to allocate device pointer array" << std::endl;
        return 1;
    }
    
    for (int i = 0; i < numberOfCells; i++) {
        CUDA_CHECK(cudaMalloc(&d_concentrations[i], numberOfComponents * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_concentrations[i], h_original_concentrations[i], 
                              numberOfComponents * sizeof(double), cudaMemcpyHostToDevice));
        h_cellsPull[i].concentrations = d_concentrations[i];
    }
    
    CUDA_CHECK(cudaMemcpy(d_cellsPull, h_cellsPull, numberOfCells * sizeof(BasicChemistryState), 
                          cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocksPerGrid = (numberOfCells + threadsPerBlock - 1) / threadsPerBlock;
    
    SolveReactorsSystem<<<blocksPerGrid, threadsPerBlock>>>(solver, d_cellsPull, numberOfCells);
    CUDA_CHECK(cudaDeviceSynchronize());  
    CUDA_CHECK(cudaGetLastError());       

    CUDA_CHECK(cudaMemcpy(h_cellsPull_res, d_cellsPull, numberOfCells * sizeof(BasicChemistryState), 
                          cudaMemcpyDeviceToHost));
    
    for (int i = 0; i < numberOfCells; i++) {
        h_cellsPull_res[i].concentrations = (double*)malloc(numberOfComponents * sizeof(double));
        if (!h_cellsPull_res[i].concentrations) {
            std::cerr << "Failed to allocate result concentrations for cell " << i << std::endl;
            return 1;
        }
        h_cellsPull_res[i].numberOfComponents = numberOfComponents;
        
        CUDA_CHECK(cudaMemcpy(h_cellsPull_res[i].concentrations, d_concentrations[i],
                              numberOfComponents * sizeof(double), cudaMemcpyDeviceToHost));
    }

    double maxError = 0.0;
    for (int i = 0; i < numberOfCells; i++) {
        for (int j = 0; j < numberOfComponents; j++) {
            double expected = coef * i; 
            double actual = h_cellsPull_res[i].concentrations[j];
            maxError = fmax(maxError, fabs(actual - expected));
        }
    }

    std::cout << "Number of cells: " << numberOfCells << std::endl;
    std::cout << "Max error: " << maxError << std::endl;

    for (int i = 0; i < numberOfCells; i++) {
        free(h_original_concentrations[i]);
        free(h_cellsPull_res[i].concentrations);
        CUDA_CHECK(cudaFree(d_concentrations[i]));
    }
    free(h_cellsPull);
    free(h_cellsPull_res);
    free(d_concentrations);
    free(h_original_concentrations);
    CUDA_CHECK(cudaFree(d_cellsPull));

    return 0;
}