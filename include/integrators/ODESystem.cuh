
#ifndef ODESystem_H
#define ODESystem_H

#pragma once

namespace kodes 
{
class ODESystem
{

public:

    // Constructors

        //- Construct null
        ODESystem()
        {}


    //- Destructor
    virtual ~ODESystem() = default;


    // Member Functions

        //- Return the number of equations in the system
        virtual size_t nEqns() const = 0;

        //- Calculate the derivatives in dydx
        virtual void derivatives
        (
            const double x,
            const std::vector<double>& y,
            std::vector<double>& dydx
        ) const = 0;

        //- Calculate the Jacobian of the system
        //  Need by the stiff-system solvers
        virtual void jacobian
        (
            const double x,
            const std::vector<double>& y,
            std::vector<double>& dfdx,
            std::vector<std::vector<double>>& dfdy
        ) const = 0;
};
}

#endif
