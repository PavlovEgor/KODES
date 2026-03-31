#pragma once 

typedef struct {
    int numberOfComponents;
    double* concentrations;
} BasicChemistryState;

typedef struct {
    BasicChemistryState base;
    double temperature;
} ChemistryStateT;

typedef struct {
    BasicChemistryState base;
    double pressure;
} ChemistryStateP;

typedef struct {
    BasicChemistryState base;
    double temperature;
    double pressure;
} ChemistryStateTP;


cudaError_t allocateBasicChemistryStateDevice(
    BasicChemistryState** d_state,
    int numberOfComponents
);

cudaError_t copyBasicChemistryStateToDevice(
    BasicChemistryState* d_dest,
    const BasicChemistryState* h_src
);

cudaError_t freeBasicChemistryStateDevice(
    BasicChemistryState* d_state
);

