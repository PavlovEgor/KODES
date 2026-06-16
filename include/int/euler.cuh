#ifndef euler_H
#define euler_H

#pragma once

#include "kodes.cuh"

namespace kodes 
{
class euler
:
    public kodes
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
    __host__ __device__
    euler
    (
        const ODESystem& ode, 
        const kodes::Config& config,
        std::vector<std::vector<double>>& data;  
    );


    //- Destructor
    __host__ __device__
    virtual ~euler() = default;


    // Member Functions

        //- Solve a single step dx and return the error
        __host__ __device__
        double solve
        (
            const double x0,
            const std::vector<double>& y0,
            const std::vector<double>& dydx0,
            const double dx,
            std::vector<double>& y
        ) const;
        
        //- Solve the ODE system and the update the state
        __host__ __device__
        void solve () const override;
}; 
}

#endif