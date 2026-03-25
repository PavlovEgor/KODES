#include "ReactorsSystem.H"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>

// Подключаем Cantera C++ интерфейс
#include <cantera/base/Solution.h>
#include <cantera/zerodim.h>
#include <cantera/thermo.h>
#include <cantera/kinetics.h>

#define N 10

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "=== Chemical Kinetics Validation Test ===" << std::endl;
    std::cout << "========================================" << std::endl << std::endl;
    std::cout << "Number of reactors: " << N << std::endl;

    const std::string cantera_file = "gri30.yaml";
    const double dt = 1.0e-6;
    const size_t n_cells = N;

    std::cout << "\nConfiguration:" << std::endl;
    std::cout << "  Cantera file: " << cantera_file << std::endl;
    std::cout << "  Number of cells: " << n_cells << std::endl;
    std::cout << "  Time step: " << dt << " s" << std::endl;

    std::cout << "\n--- Initialization ---" << std::endl;

    ReactorsSystem combustion(
        N,
        "gri30.yaml",
        "RK4.json"
    );
    size_t n_species = combustion.getNumSpecies();
    std::cout << "Number of species: " << n_species << std::endl;
    
    CanteraValidator validator(cantera_file);
    if (n_species != validator.getNumSpecies()) {
        std::cerr << "ERROR: Species count mismatch!" << std::endl;
        return 1;
    }

    
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
    
    std::cout << "\nInitial conditions (first 3 cells):" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    
    for (size_t cell = 0; cell < std::min(size_t(3), N); ++cell) {
        std::cout << "\nCell " << cell << ":" << std::endl;
        std::cout << "  Temperature: " << temperature[cell] << " K" << std::endl;
        std::cout << "  Pressure:    " << pressure[cell] << " Pa" << std::endl;
        std::cout << "  Density:     " << density[cell] << " kg/m³" << std::endl;
        std::cout << "  Concentrations (first 5 species): ";
        
        double sum_check = 0.0;
        for (size_t sp = 0; sp < std::min(size_t(5), n_species); ++sp) {
            std::cout << concentrations[sp][cell] << " ";
            sum_check += concentrations[sp][cell];
        }
        std::cout << std::endl;
        std::cout << "  Sum of concentrations (all species): " << sum_check << std::endl;
    }
    
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
    
    double dt = 1.0e-6;  // 1 микросекунда
    std::cout << "\nPerforming one chemistry step with dt = " << dt << " s..." << std::endl;
    
    auto solve_start = std::chrono::high_resolution_clock::now();
    
    combustion.solve(dt);
    
    auto solve_end = std::chrono::high_resolution_clock::now();
    double solve_time = std::chrono::duration<double, std::milli>(solve_end - solve_start).count();
    std::cout << "Solve time: " << solve_time << " ms" << std::endl;
    
    std::cout << "\nRetrieving results from GPU..." << std::endl;
    
    auto download_start = std::chrono::high_resolution_clock::now();
    
    std::vector<std::vector<double>> new_concentrations;
    combustion.getConcentrations(new_concentrations);
    
    auto download_end = std::chrono::high_resolution_clock::now();
    double download_time = std::chrono::duration<double, std::milli>(download_end - download_start).count();
    std::cout << "Results retrieval time: " << download_time << " ms" << std::endl;
    
    std::cout << "\n=== Results Analysis ===" << std::endl;
    
    double max_change = 0.0;
    double avg_change = 0.0;
    int changes_count = 0;
    
    for (size_t cell = 0; cell < std::min(size_t(3), N); ++cell) {
        std::cout << "\nCell " << cell << " concentration changes:" << std::endl;
        
        for (size_t sp = 0; sp < std::min(size_t(5), n_species); ++sp) {
            double initial = concentrations[sp][cell];
            double final = new_concentrations[sp][cell];
            double change = std::abs(final - initial);
            
            if (change > max_change) max_change = change;
            avg_change += change;
            changes_count++;
            
            std::cout << "  " << std::setw(12) << species_names[sp] 
                      << ": " << std::setw(10) << initial 
                      << " -> " << std::setw(10) << final 
                      << " (Δ = " << change << ")" << std::endl;
        }
    }
    
    avg_change /= changes_count;
    
    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "Maximum concentration change: " << max_change << std::endl;
    std::cout << "Average concentration change: " << avg_change << std::endl;
    std::cout << "Total time (transfer + solve + retrieval): " 
              << transfer_time + solve_time + download_time << " ms" << std::endl;
    
    // 12. Проверка суммы концентраций (должна сохраняться)
    std::cout << "\n=== Conservation Check ===" << std::endl;
    
    double max_sum_error = 0.0;
    for (size_t cell = 0; cell < N; ++cell) {
        double sum_initial = 0.0;
        double sum_final = 0.0;
        
        for (size_t sp = 0; sp < n_species; ++sp) {
            sum_initial += concentrations[sp][cell];
            sum_final += new_concentrations[sp][cell];
        }
        
        double error = std::abs(sum_final - sum_initial);
        if (error > max_sum_error) max_sum_error = error;
        
        if (cell < 3) {
            std::cout << "Cell " << cell << ": sum_initial = " << sum_initial 
                      << ", sum_final = " << sum_final 
                      << ", error = " << error << std::endl;
        }
    }
    std::cout << "Maximum sum error: " << max_sum_error << std::endl;
    
    // 13. Получение статистики производительности (если реализовано)
    auto stats = combustion.getPerformanceStats();
    std::cout << "\n=== Performance Statistics ===" << std::endl;
    std::cout << "Solve calls: " << stats.num_calls << std::endl;
    std::cout << "Average solve time: " << stats.total_solve_time_ms / stats.num_calls << " ms" << std::endl;
    
    std::cout << "\nTest completed successfully!" << std::endl;
    
    return 0;
}