#include "euler.cuh"
#include <algorithm>
#include <iostream>
#include <cmath>


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

kodes::euler::euler(const ODESystem& ode, const kodes::Config& config)
:
    integrator(ode, config),
    safeScale_(config.getDouble("safeScale", 0.9)),
    alphaInc_(config.getDouble("alphaIncrease", 0.2)),
    alphaDec_(config.getDouble("alphaDecrease", 0.25)),
    minScale_(config.getDouble("minScale", 0.2)),
    maxScale_(config.getDouble("maxScale", 10)),
    dydx0_(ode.nEqns()),
    yTemp_(ode.nEqns()),
    err_(ode.nEqns())
{
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

double kodes::euler::solve
(
    const double x0,
    const std::vector<double>& y0,
    const std::vector<double>& dydx0,
    const double dx,
    std::vector<double>& y
) const
{
    // Calculate error estimate from the change in state:
    for(size_t i=0; i < err_.size(); ++i)
    {
        err_[i] = dx*dydx0[i];
    }
    for(size_t i=0; i < err_.size(); ++i)
    {
        y[i] = y0[i] + err_[i];
    }

    return normalizeError(y0, y, err_);
}

void kodes::euler::solve
(
    double& x,
    std::vector<double>& y,
    double& dxTry
) const
{
    double dx = dxTry;
    double err = 0.0;

    odes_.derivatives(x, y, dydx0_);

    // Loop over solver and adjust step-size as necessary
    // to achieve desired error
    do
    {
        // Solve step and provide error estimate
        err = solve(x, y, dydx0_, dx, yTemp_);

        // If error is large reduce dx
        if (err > 1)
        {
            double scale = max(safeScale_*pow(err, -alphaDec_), minScale_);
            dx *= scale;

            // if (dx < VSMALL)
            // {
            //     FatalErrorInFunction
            //         << "stepsize underflow"
            //         << exit(FatalError);
            // }
        }
    } while (err > 1);

    // Update the state
    x += dx;
    y = yTemp_;

    // If the error is small increase the step-size
    if (err > pow(maxScale_/safeScale_, -1.0/alphaInc_))
    {
        double scale = safeScale_*pow(err, -alphaInc_);
        dxTry = std::clamp(scale, minScale_, maxScale_)*dx;
    }
    else
    {
        dxTry = safeScale_*maxScale_*dx;
    }
}