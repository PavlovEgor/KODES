// SimpleSystem.cpp
#include "SimpleSystem.cuh"

namespace kodes 
{

SimpleSystem::SimpleSystem()
{
}

SimpleSystem::~SimpleSystem()
{
}

size_t SimpleSystem::nEqns() const
{
    return 1;
}

void SimpleSystem::derivatives
(
    const double x,
    const std::vector<double>& y,
    std::vector<double>& dydx
) const
{
    dydx[0] = - y[0];
}

void SimpleSystem::jacobian
(
    const double x,
    const std::vector<double>& y,
    std::vector<double>& dfdx,
    std::vector<std::vector<double>>& dfdy
) const
{
    dfdy[0][0] = -1.0;
}

std::vector<double> SimpleSystem::getInitialConditions(){
    return std::vector<double>(1, 1.0);
}

std::vector<double> SimpleSystem::getGroundSolution(double x)
{
    return std::vector<double>(1, exp(-x));
}

} // namespace kodes