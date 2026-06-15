
// PollutionSystem.h
#ifndef PollutionSystem_H
#define PollutionSystem_H

#pragma once

#include "ODESystem.cuh"
#include <vector>

namespace kodes 
{

class PollutionSystem : public ODESystem
{
private:

    const std::vector<double> k =
    {
        0.350,           // k1
        0.266e2,         // k2
        0.123e5,         // k3
        0.860e-3,        // k4
        0.820e-3,        // k5
        0.150e5,         // k6
        0.130e-3,        // k7
        0.240e5,         // k8
        0.165e5,         // k9
        0.900e4,         // k10
        0.220e-1,        // k11
        0.120e5,         // k12
        0.188e1,         // k13
        0.163e5,         // k14
        0.480e7,         // k15
        0.350e-3,        // k16
        0.175e-1,        // k17
        0.100e9,         // k18
        0.444e12,        // k19
        0.124e4,         // k20
        0.210e1,         // k21
        0.578e1,         // k22
        0.474e-1,        // k23
        0.178e4,         // k24
        0.312e1          // k25
    };

public:
    // Constructor
    PollutionSystem();
    
    // Destructor
    ~PollutionSystem();
    
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

#endif // PollutionSystem_H
