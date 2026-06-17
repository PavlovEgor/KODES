#ifndef seulex_H
#define seulex_H

#pragma once

#include "integrator.cuh"

namespace kodes 
{
class seulex
:
    public integrator
{
    // Private data

        // Static constants

            static const size_t kMaxx_ = 12;
            static const size_t iMaxx_ = kMaxx_ + 1;

            static const double
                stepFactor1_, stepFactor2_, stepFactor3_,
                stepFactor4_, stepFactor5_,
                kFactor1_, kFactor2_;

        // Evaluated constants

            double jacRedo_;
            std::vector<int> nSeq_;
            std::vector<double> cpu_;
            std::vector<std::vector<double>> coeff_;

        // Temporary storage
        // held to avoid dynamic memory allocation between calls
        // and to transfer internal values between functions

            mutable double theta_;
            mutable size_t kTarg_;
            mutable std::vector<std::vector<double>> table_;

            mutable std::vector<double> dfdx_;
            mutable std::vector<std::vector<double>> dfdy_;
            mutable std::vector<std::vector<double>> a_;
            mutable std::vector<int> pivotIndices_;

            // Fields space for "solve" function
            mutable std::vector<double> dxOpt_, temp_;
            mutable std::vector<double> y0_, ySequence_, scale_;

            // Fields used in "seul" function
            mutable std::vector<double> dy_, yTemp_, dydx_;


    // Private Member Functions

        //- Computes the j-th line of the extrapolation table
        bool seul
        (
            const double x0,
            const std::vector<double>& y0,
            const double dxTot,
            const size_t k,
            std::vector<double>& y,
            const std::vector<double>& scale
        ) const;

        //- Polynomial extrpolation
        void extrapolate
        (
            const size_t k,
            std::vector<std::vector<double>>& table,
            std::vector<double>& y
        ) const;


public:

    // Constructors

        //- Construct from ODESystem
        seulex(const ODESystem& ode, const kodes::Config& config);


    //- Destructor
    virtual ~seulex() = default;


    // Member Functions

        using integrator::solve;

        //- Solve the ODE system and the update the state
        void solve 
        (
            double& x,
            std::vector<double>& y,
            stepState& step
        ) const override;
}; 
}

template <typename T>
inline
T sqr(const T& x)
{
    return x * x;
}


#endif