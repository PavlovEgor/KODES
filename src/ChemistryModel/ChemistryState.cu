#include "ChemistryState.cuh"

cudaError_t allocateBasicChemistryStateDevice(
    BasicChemistryState** d_state,
    int numberOfComponents
) {
    cudaError_t err = cudaSuccess;
    
    err = cudaMalloc(d_state, sizeof(BasicChemistryState));
    if (err != cudaSuccess) return err;
    
    double* d_concentrations = nullptr;
    err = cudaMalloc(&d_concentrations, numberOfComponents * sizeof(double));
    if (err != cudaSuccess) {
        cudaFree(*d_state);
        return err;
    }
    
    BasicChemistryState temp_state;
    temp_state.numberOfComponents = numberOfComponents;
    temp_state.concentrations = d_concentrations;
    
    err = cudaMemcpy(*d_state, &temp_state, sizeof(BasicChemistryState), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_concentrations);
        cudaFree(*d_state);
    }
    
    return err;
}

cudaError_t copyBasicChemistryStateToDevice(
    BasicChemistryState* d_dest,
    const BasicChemistryState* h_src
) {
    BasicChemistryState d_temp;
    cudaError_t err = cudaMemcpy(&d_temp, d_dest, sizeof(BasicChemistryState), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return err;
    
    size_t array_size = h_src->numberOfComponents * sizeof(double);
    err = cudaMemcpy(
        d_temp.concentrations,
        h_src->concentrations,
        array_size,
        cudaMemcpyHostToDevice
    );
    if (err != cudaSuccess) return err;

    return err;
}

cudaError_t freeBasicChemistryStateDevice(BasicChemistryState* d_state) {
    if (d_state == nullptr) return cudaSuccess;
    
    BasicChemistryState temp;
    cudaError_t err = cudaMemcpy(&temp, d_state, sizeof(BasicChemistryState), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return err;
    
    if (temp.concentrations != nullptr) {
        err = cudaFree(temp.concentrations);
        if (err != cudaSuccess) return err;
    }
    
    err = cudaFree(d_state);
    
    return err;
}