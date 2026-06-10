#include "integrator.cuh"
#include <cmath>
#include <cstdio>
#include <stdexcept>

// * * * * * * * * * * * * Protected Member Functions  * * * * * * * * * * * //

double kodes::integrator::normalizeError
(
    const std::vector<double>& y0,
    const std::vector<double>& y,
    const std::vector<double>& err
) const
{
    // Calculate the maximum error
    double maxErr = 0.0;
    forAll(err, i)
    {
        double tol = absTol_[i] + relTol_[i]*max(fabs(y0[i]), fabs(y[i]));
        maxErr = max(maxErr, fabs(err[i])/tol);
    }

    return maxErr;
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

kodes::integrator::integrator(const ODESystem& ode, const kodes::Config& config)
:
    odes_(ode),
    sizeOfSystem_(ode.nEqns()),
    absTol_(sizeOfSystem_, config.getDouble("absTol", 1e-4)),
    relTol_(sizeOfSystem_, config.getDouble("relTol", 1e-4)),
    maxSteps_(config.getInt("maxSteps", 10000))
{
}


kodes::integrator::integrator
(
    const ODESystem& ode,
    const std::vector<double>& absTol,
    const std::vector<double>& relTol
)
:
    odes_(ode),
    sizeOfSystem_(ode.nEqns()),
    absTol_(absTol),
    relTol_(relTol),
    maxSteps_(10000)
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void kodes::integrator::solve
(
    double& x,
    std::vector<double>& y,
    double& dxTry
) const
{
    stepState step(dxTry);
    solve(x, y, step);
    dxTry = step.dxTry;
}


void kodes::integrator::solve
(
    double& x,
    std::vector<double>& y,
    stepState& step
) const
{
    double x0 = x;
    solve(x, y, step.dxTry);
    step.dxDid = x - x0;
}


void kodes::integrator::solve
(
    const double xStart,
    const double xEnd,
    std::vector<double>& y,
    double& dxTry
) const
{
    stepState step(dxTry);
    double x = xStart;

    for (size_t nStep=0; nStep<maxSteps_; ++nStep)
    {
        // Store previous iteration dxTry
        double dxTry0 = step.dxTry;

        step.reject = false;

        // Check if this is a truncated step and set dxTry to integrate to xEnd
        if ((x + step.dxTry - xEnd)*(x + step.dxTry - xStart) > 0)
        {
            step.last = true;
            step.dxTry = xEnd - x;
        }

        // Integrate as far as possible up to step.dxTry
        solve(x, y, step);

        // Check if reached xEnd
        if ((x - xEnd)*(xEnd - xStart) >= 0)
        {
            if (nStep > 0 && step.last)
            {
                step.dxTry = dxTry0;
            }

            dxTry = step.dxTry;

            return;
        }

        step.first = false;

        // If the step.dxTry was reject set step.prevReject
        if (step.reject)
        {
            step.prevReject = true;
        }
    }
}


// ************************************************************************* //