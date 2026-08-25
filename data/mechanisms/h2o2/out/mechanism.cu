#include "mass_mole.cuh"
#include <stdio.h>
#include "mechanism.cuh"
#include "gpu_memory.cuh"
    //apply masking of ICs for cache optimized mechanisms
    void apply_mask(double* y_specs) {
    }
    //reverse masking of ICs for cache optimized mechanisms
    void apply_reverse_mask(double* y_specs) {
    }
void set_same_initial_conditions(int NUM, double** y_host, double** var_host) 
{
    double Xi [NSP] = {0.0};
    //set initial mole fractions here

    Xi[0]   =   2;  // H2
    Xi[3]   =   1;  // O2

    //Normalize mole fractions to sum to one
    double Xsum = 0.0;
    for (int j = 0; j < NSP; ++ j) {
        Xsum += Xi[j];
    }
    if (Xsum == 0.0) {
        printf("Use of the set initial conditions function requires user implementation!\n");
        exit(-1);
    }
    for (int j = 0; j < NSP; ++ j) {
        Xi[j] /= Xsum;
    }

    //convert to mass fractions
    double Yi[NSP - 1] = {0.0};
    mole2mass(Xi, Yi);

    //set initial pressure, units [PA]
    double P = 101325.0;
    // set intial temperature, units [K]
    double T0 = 2000;

    // (*y_host) = (double*)malloc(NUM * NSP * sizeof(double));
    var_host[0] = (double*)malloc(NUM * sizeof(double));
    //load temperature and mass fractions for all threads (cells)
    for (int i = 0; i < NSP; ++i) {
        y_host[i] = (double*)malloc(NUM * sizeof(double));
        
        for (int j=0; j<NUM; ++j)
        {
            if (i == 0)
            {
                y_host[i][j] = T0;
            }
            else
            {
                y_host[i][j] = Yi[i-1];
            }
        }
    }

#ifdef CONV
    //calculate density
    double rho = getDensity(T0, P, Xi);
#endif

    for (int i = 0; i < NUM; ++i) {
#ifdef CONV
        var_host[0][i] = rho;
#elif defined(CONP)
        var_host[0][i] = P;
#endif
    }
}

