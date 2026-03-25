// ReactorsSystem.h
#pragma once

#include <vector>
#include <memory>
#include <string>

#include "ChemistryModel.H"
#include "Reactor.H"
#include "Solver.H"

#include "ThermodynamicData.H"
#include "TransportData.H"
#include "ReactionsData.H"
#include "SolverConfig.H"

namespace kodes {

class ReactorsSystem {
public:

    ReactorsSystem(
        size_t n_cells,
        const vector<double>& cells_vol;
        const std::string& cantera_file,
        const std::string& solver_config_file,
        int gpu_device_id = 0
    );
    

    ~ReactorsSystem();
    
    ReactorsSystem(const ReactorsSystem&) = delete;
    ReactorsSystem& operator=(const ReactorsSystem&) = delete;
    
    ReactorsSystem(ReactorsSystem&& other) noexcept;
    ReactorsSystem& operator=(ReactorsSystem&& other) noexcept;
    
    void setInitialState(
        const std::vector<std::vector<double>>& concentrations,  // [n_species][n_cells]
        const std::vector<double>& temperature,                  // [n_cells]
        const std::vector<double>& pressure,                     // [n_cells]
        const std::vector<double>& density                       // [n_cells]
    );
    
    void solve(double dt);
    
    void getConcentrations(
        std::vector<std::vector<double>>& concentrations
    ) const;
    
    
private:
    
    size_t n_cells_;
    size_t n_species_;
    
    Solver              solver_;
    ChemistryModel      chemistry_model_;
};

} 