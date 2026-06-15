#ifndef euler_H
#define euler_H

#pragma once

#include "integrator.cuh"

namespace kodes 
{
class euler
:
    public integrator
{
    // Private data
    mutable std::vector<double> err_;

    //- Step-size adjustment controls
    double safeScale_, alphaInc_, alphaDec_, minScale_, maxScale_;

    //- Cache for dydx at the initial time
    mutable std::vector<double> dydx0_;

    //- Temporary for the test-step solution
    mutable std::vector<double> yTemp_;

public:

    // Constructors

        //- Construct from ODESystem
        euler(const ODESystem& ode, const kodes::Config& config);


    //- Destructor
    virtual ~euler() = default;


    // Member Functions

        using integrator::solve;

        //- Solve a single step dx and return the error
        double solve
        (
            const double x0,
            const std::vector<double>& y0,
            const std::vector<double>& dydx0,
            const double dx,
            std::vector<double>& y
        ) const;
        
        //- Solve the ODE system and the update the state
        void solve 
        (
            double& x,
            std::vector<double>& y,
            double& dxTry
        ) const override;
}; 
}

#endif