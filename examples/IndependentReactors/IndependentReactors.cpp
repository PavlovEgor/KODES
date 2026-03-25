#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include "math.h"


#include "ReactorsSystem.H"

#include "cantera/zerodim.h"

#define N 10


void kodesSolution( std::vector<std::vector<double>>& new_concentrations,
                    const std::vector<std::vector<double>>& old_concentrations, 
                    const std::vector<double>& temperature,                 
                    const std::vector<double>& pressure,                     
                    const std::vector<double>& density )
{
    const std::string cantera_file = "gri30.yaml";
    const double dt = 1.0e-2;
    const size_t n_cells = N;

    ReactorsSystem combustion(
        N,
        cantera_file,
        "RK4.json"
    );

    size_t n_species = combustion.getNumSpecies();

        std::cout << "\nTransferring data to GPU..." << std::endl;
    
    auto transfer_start = std::chrono::high_resolution_clock::now();
    
    combustion.setInitialState(
        concentrations,
        temperature,
        pressure,
        density
    );
    
    auto transfer_end = std::chrono::high_resolution_clock::now();
    double transfer_time = std::chrono::duration<double, std::milli>(transfer_end - transfer_start).count();
    std::cout << "Data transfer time: " << transfer_time << " ms" << std::endl;
    
    std::cout << "\nPerforming one chemistry step with dt = " << dt << " s..." << std::endl;
    
    auto solve_start = std::chrono::high_resolution_clock::now();
    
    combustion.solve(dt);
    
    auto solve_end = std::chrono::high_resolution_clock::now();
    double solve_time = std::chrono::duration<double, std::milli>(solve_end - solve_start).count();
    std::cout << "Solve time: " << solve_time << " ms" << std::endl;
    
    std::cout << "\nRetrieving results from GPU..." << std::endl;
    
    auto download_start = std::chrono::high_resolution_clock::now();
    
    combustion.getConcentrations(new_concentrations);
    
    auto download_end = std::chrono::high_resolution_clock::now();
    double download_time = std::chrono::duration<double, std::milli>(download_end - download_start).count();
    std::cout << "Results retrieval time: " << download_time << " ms" << std::endl;

}

std::vector<double> massToMoleFractions(const std::vector<double>& massFractions,
                                        const std::vector<double>& molecularWeights)
{
    size_t n_species = massFractions.size();
    std::vector<double> moleFractions(n_species, 0.0);
    
    double sum = 0.0;
    for (size_t i = 0; i < n_species; i++) {
        if (massFractions[i] > 0 && molecularWeights[i] > 0) {
            moleFractions[i] = massFractions[i] / molecularWeights[i];
            sum += moleFractions[i];
        }
    }
    
    if (sum > 0) {
        for (size_t i = 0; i < n_species; i++) {
            moleFractions[i] /= sum;
        }
    }
    
    return moleFractions;
}
std::vector<double> moleToMassFractions(const std::vector<double>& moleFractions,
                                        const std::vector<double>& molecularWeights)
{
    size_t n_species = moleFractions.size();
    std::vector<double> massFractions(n_species, 0.0);
    
    double sum = 0.0;
    for (size_t i = 0; i < n_species; i++) {
        if (moleFractions[i] > 0 && molecularWeights[i] > 0) {
            massFractions[i] = moleFractions[i] * molecularWeights[i];
            sum += massFractions[i];
        }
    }
    
    if (sum > 0) {
        for (size_t i = 0; i < n_species; i++) {
            massFractions[i] /= sum;
        }
    }
    
    return massFractions;
}
std::vector<double> getMolecularWeights(shared_ptr<ThermoPhase> gas)
{
    size_t n_species = gas->nSpecies();
    std::vector<double> molWeights(n_species);
    
    for (size_t i = 0; i < n_species; i++) {
        molWeights[i] = gas->molecularWeight(i);
    }
    
    return molWeights;
}
void canteraSolution( std::vector<std::vector<double>>& new_concentrations,
                    const std::vector<std::vector<double>>& old_concentrations, 
                    const std::vector<double>& temperature,                 
                    const std::vector<double>& pressure,                     
                    const std::vector<double>& density )
{
    auto sol = newSolution("gri30.yaml", "gri30", "none");
    auto gas = sol->thermo();
    std::vector<double> molWeights = getMolecularWeights(gas);
    const size_t n_cells = N;

    auto exhaust = newReservoir(sol);
    for (size_t cell = 0; cell < n_cells; cell++) {
        try {
            double T = temperature[cell];
            double P = pressure[cell];
            double rho = density[cell];
            
            std::string composition;
            std::vector<double> moleFractions = massToMoleFractions(old_mass_fractions[idx], molWeights);
            
            for (size_t i = 0; i < n_species && i < old_concentrations[cell].size(); i++) {
                if (moleFractions[i] > 1e-20) {
                    if (!composition.empty()) composition += ", ";
                    composition += gas->speciesName(i) + ":" + std::to_string(moleFractions);
                }
            }
            
            gas->setState_TPX(T, P, composition);
            
            auto reactor = newReactor4("Reactor", sol);
            reactor->setInitialVolume(1.0); // Объем 1 м^3
            
            ReactorNet sim(reactor);
            
            double dt = 1e-2;     
            sim.advance(dt);
            
            auto final_gas = reactor->phase()->thermo();
            std::vector<double> final_moleFractions(n_species);
            final_gas->getMoleFractions(final_moleFractions.data());
            new_concentrations[cell] = moleToMassFractions(final_moleFractions, molWeights);
            
        } catch (CanteraError& err) {
            std::cout << "Error in case " << cell << ": " << err.what() << std::endl;
            for (size_t i = 0; i < n_species; i++) {
                new_concentrations[cell][i] = 0.0;
            }
        }
    }
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "=== Chemical Kinetics Validation Test ===" << std::endl;
    std::cout << "========================================" << std::endl << std::endl;
    std::cout << "Number of reactors: " << N << std::endl;

    
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> pressure_dist(80000.0, 120000.0);
    std::uniform_real_distribution<double> temperature_dist(280.0, 320.0);
    std::uniform_real_distribution<double> density_dist(1.0, 1.4);
    std::uniform_real_distribution<double> conc_dist(0.0, 1.0);

    std::vector<std::vector<double>> concentrations(n_species, std::vector<double>(N));
    std::vector<double> temperature(N);
    std::vector<double> pressure(N);
    std::vector<double> density(N);
    
    std::cout << "Generating random initial conditions..." << std::endl;
    
    for (size_t cell = 0; cell < N; ++cell) {
        double sum = 0.0;
        
        for (size_t sp = 0; sp < n_species; ++sp) {
            concentrations[sp][cell] = conc_dist(rng);
            sum += concentrations[sp][cell];
        }
        
        for (size_t sp = 0; sp < n_species; ++sp) {
            concentrations[sp][cell] = concentrations[sp][cell] / sum;
        }
        
        temperature[cell] = temperature_dist(rng);
        pressure[cell] = pressure_dist(rng);
        density[cell] = density_dist(rng);
    }
    

    std::vector<std::vector<double>> kodes_concentrations(n_species, std::vector<double>(N));
    std::vector<std::vector<double>> cantera_concentrations(n_species, std::vector<double>(N));
    
    kodesSolution(kodes_concentrations,
                concentrations,
                temperature,
                pressure,
                density);
    
    canteraSolution(cantera_concentrations,
                concentrations,
                temperature,
                pressure,
                density);
    

    for(int i=0, i<N, i++){
        for (int j=0, j<n_species, j++){
            if (fabs(kodes_concentrations[j][i] - cantera_concentrations[j][i]) > 1e-6){
                std::cout << "\nTest failed!" << std::endl;
            }
        }
    }
    std::cout << "\nTest completed successfully!" << std::endl;
    
    return 0;
}