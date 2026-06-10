
#ifndef integrator_H
#define integrator_H

#pragma once

#include "ODESystem.cuh"

#include <string>
#include "kodes_config.cuh"


namespace kodes 
{

class integrator
{
protected:

    const ODESystem& odes_;
    
    const size_t sizeOfSystem_;

    std::vector<double> absTol_;
    std::vector<double> relTol_;

    size_t maxSteps_; 

    double normalizeError
    (
        const std::vector<double>& y0,
        const std::vector<double>& y,
        const std::vector<double>& err
    ) const;

    //- No copy construct
    integrator(const integrator&) = delete;

    //- No copy assignment
    void operator=(const integrator&) = delete;

public:

    friend class ODESystem; 

    class stepState
    {
    public:

        const bool forward;
        double dxTry;
        double dxDid;
        bool first;
        bool last;
        bool reject;
        bool prevReject;

        stepState(const double dx)
        :
            forward(dx > 0 ? true : false),
            dxTry(dx),
            dxDid(0),
            first(true),
            last(false),
            reject(false),
            prevReject(false)
        {}
    }; 

    // Constructors

        //- Construct for given ODESystem
        integrator(const ODESystem& ode, const kodes::Config& config);

        //- Construct for given ODESystem specifying tolerances
        integrator
        (
            const ODESystem& ode,
            const std::vector<double>& absTol,
            const std::vector<double>& relTol
        );

    //- Destructor
    virtual ~integrator() = default;

    // Member Functions

        //- The number of equations to solve
        size_t nEqns() const noexcept { return sizeOfSystem_; }

        //- Access to the absolute tolerance field
        std::vector<double>& absTol() noexcept { return absTol_; }

        //- Access to the relative tolerance field
        std::vector<double>& relTol() noexcept { return relTol_; }

        //- Solve the ODE system as far as possible up to dxTry
        //  adjusting the step as necessary to provide a solution within
        //  the specified tolerance.
        //  Update the state and return an estimate for the next step in dxTry
        virtual void solve
        (
            double& x,
            std::vector<double>& y,
            double& dxTry
        ) const;

        //- Solve the ODE system as far as possible up to dxTry
        //  adjusting the step as necessary to provide a solution within
        //  the specified tolerance.
        //  Update the state and return an estimate for the next step in dxTry
        virtual void solve
        (
            double& x,
            std::vector<double>& y,
            stepState& step
        ) const;

        //- Solve the ODE system from xStart to xEnd, update the state
        //  and return an estimate for the next step in dxTry
        virtual void solve
        (
            const double xStart,
            const double xEnd,
            std::vector<double>& y,
            double& dxEst
        ) const;
};

}

#endif
