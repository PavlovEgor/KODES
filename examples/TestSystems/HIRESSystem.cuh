
// HIRESSystem.h
#ifndef HIRESSystem_H
#define HIRESSystem_H

#pragma once

#include "ODESystem.cuh"
#include <vector>

namespace kodes 
{

class HIRESSystem : public ODESystem
{
public:
    // Constructor
    HIRESSystem();
    
    // Destructor
    ~HIRESSystem();
    
    // Return the number of equations in the system
    size_t nEqns() const override;
    
    // Calculate the derivatives in dydx
    void derivatives
    (
        const double x,
        const std::vector<double>& y,
        std::vector<double>& dydx
    ) const override;
    
    // Calculate the Jacobian of the system
    void jacobian
    (
        const double x,
        const std::vector<double>& y,
        std::vector<double>& dfdx,
        std::vector<std::vector<double>>& dfdy
    ) const override;
    
    // Helper function to set initial conditions
    static std::vector<double> getInitialConditions();
    
    // approximate solution in point 321.8122
    static std::vector<double> getGroundSolution();
};

} // namespace kodes

#endif // HIRESSystem_H
