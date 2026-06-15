// HIRESSystem.cpp
#include "HIRESSystem.cuh"

namespace kodes 
{

HIRESSystem::HIRESSystem()
{
}

HIRESSystem::~HIRESSystem()
{
}

size_t HIRESSystem::nEqns() const
{
    return 8;
}

void HIRESSystem::derivatives
(
    const double x,
    const std::vector<double>& y,
    std::vector<double>& dydx
) const
{
    double y1 = y[0];
    double y2 = y[1];
    double y3 = y[2];
    double y4 = y[3];
    double y5 = y[4];
    double y6 = y[5];
    double y7 = y[6];
    double y8 = y[7];
    
    // y1' = -1.71*y1 + 0.43*y2 + 8.32*y3 + 0.0007
    dydx[0] = -1.71 * y1 + 0.43 * y2 + 8.32 * y3 + 0.0007;
    
    // y2' = 1.71*y1 - 8.75*y2
    dydx[1] = 1.71 * y1 - 8.75 * y2;
    
    // y3' = -10.03*y3 + 0.43*y4 + 0.035*y5
    dydx[2] = -10.03 * y3 + 0.43 * y4 + 0.035 * y5;
    
    // y4' = 8.32*y2 + 1.71*y3 - 1.12*y4
    dydx[3] = 8.32 * y2 + 1.71 * y3 - 1.12 * y4;
    
    // y5' = -1.745*y5 + 0.43*y6 + 0.43*y7
    dydx[4] = -1.745 * y5 + 0.43 * y6 + 0.43 * y7;
    
    // y6' = -280*y6*y8 + 0.69*y4 + 1.71*y5 - 0.43*y6 + 0.69*y7
    dydx[5] = -280.0 * y6 * y8 + 0.69 * y4 + 1.71 * y5 - 0.43 * y6 + 0.69 * y7;
    
    // y7' = 280*y6*y8 - 1.81*y7
    dydx[6] = 280.0 * y6 * y8 - 1.81 * y7;
    
    // y8' = -y7
    dydx[7] = -280 * y6 * y8 + 1.81 * y7;
}

void HIRESSystem::jacobian
(
    const double x,
    const std::vector<double>& y,
    std::vector<double>& dfdx,
    std::vector<std::vector<double>>& dfdy
) const
{
    size_t n = nEqns();
    
    // df/dx = 0 for autonomous system
    for (size_t i = 0; i < n; ++i)
    {
        dfdx[i] = 0.0;
    }
    
    // Initialize Jacobian matrix with zeros
    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j < n; ++j)
        {
            dfdy[i][j] = 0.0;
        }
    }
    
    double y6 = y[5];
    double y8 = y[7];
    
    // Row 0: derivatives of y1'
    dfdy[0][0] = -1.71;
    dfdy[0][1] = 0.43;
    dfdy[0][2] = 8.32;
    
    // Row 1: derivatives of y2'
    dfdy[1][0] = 1.71;
    dfdy[1][1] = -8.75;
    
    // Row 2: derivatives of y3'
    dfdy[2][2] = -10.03;
    dfdy[2][3] = 0.43;
    dfdy[2][4] = 0.035;
    
    // Row 3: derivatives of y4'
    dfdy[3][1] = 8.32;
    dfdy[3][2] = 1.71;
    dfdy[3][3] = -1.12;
    
    // Row 4: derivatives of y5'
    dfdy[4][4] = -1.745;
    dfdy[4][5] = 0.43;
    dfdy[4][6] = 0.43;
    
    // Row 5: derivatives of y6'
    dfdy[5][3] = 0.69;
    dfdy[5][4] = 1.71;
    dfdy[5][5] = -280.0 * y8 - 0.43;
    dfdy[5][6] = 0.69;
    dfdy[5][7] = -280.0 * y6;
    
    // Row 6: derivatives of y7'
    dfdy[6][5] = 280.0 * y8;
    dfdy[6][6] = -1.81;
    dfdy[6][7] = 280.0 * y6;
    
    // Row 7: derivatives of y8'
    dfdy[7][5] = -280*y8;
    dfdy[7][6] = 1.81;
    dfdy[7][7] = -280 * y6;


}

std::vector<double> HIRESSystem::getInitialConditions()
{
    std::vector<double> y0(8, 0.0);
    y0[0] = 1.0;      // y1(0) = 1
    y0[7] = 0.0057;   // y8(0) = 0.0057
    // y2(0) through y7(0) are 0
    return y0;
}

std::vector<double> HIRESSystem::getGroundSolution()
{
    std::vector<double> yGround(8, 0.0);
    yGround[0] = 0.7371312573325668e-3, yGround[1] = 0.1442485726316185e-3;
    yGround[2] = 0.5888729740967575e-4, yGround[3] = 0.1175651343283149e-2;
    yGround[4] = 0.2386356198831331e-2, yGround[5] = 0.6238968252742796e-2;
    yGround[6] = 0.2849998395185769e-2, yGround[7] = 0.2850001604814231e-2;

    return yGround;
}

} // namespace kodes