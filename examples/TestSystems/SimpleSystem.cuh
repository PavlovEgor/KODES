
// SimpleSystem.h
#ifndef SimpleSystem_H
#define SimpleSystem_H

#pragma once

#include "ODESystem.cuh"
#include <vector>

namespace kodes 
{

class SimpleSystem : public ODESystem
{
public:
    // Constructor
    SimpleSystem();
    
    // Destructor
    ~SimpleSystem();
    
    // Return the number of equations in the system
    size_t nEqns() const;
    
    // Calculate the derivatives in dydx
    void derivatives
    (
        const double x,
        const std::vector<double>& y,
        std::vector<double>& dydx
    ) const;
    
    // Calculate the Jacobian of the system
    void jacobian
    (
        const double x,
        const std::vector<double>& y,
        std::vector<double>& dfdx,
        std::vector<std::vector<double>>& dfdy
    ) const;
    
    // Helper function to set initial conditions
    static std::vector<double> getInitialConditions();
    
    // Output points of interest
    static std::vector<double> getGroundSolution(double);
};

} // namespace kodes

#endif // SimpleSystem_H
