// PollutionSystem.cpp
#include "PollutionSystem.cuh"

namespace kodes 
{

PollutionSystem::PollutionSystem()
{
}

PollutionSystem::~PollutionSystem()
{
}

size_t PollutionSystem::nEqns() const
{
    return 20;
}

void PollutionSystem::derivatives
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
    double y9 = y[8];
    double y10 = y[9];
    double y11 = y[10];
    double y12 = y[11];
    double y13 = y[12];
    double y14 = y[13];
    double y15 = y[14];
    double y16 = y[15];
    double y17 = y[16];
    double y18 = y[17];
    double y19 = y[18];
    double y20 = y[19];

    double r1 = k[0] * y1;
    double r2 = k[1] * y2 * y4;
    double r3 = k[2] * y5 * y2;
    double r4 = k[3] * y7;
    double r5 = k[4] * y7;
    double r6 = k[5] * y7 * y6;
    double r7 = k[6] * y9;
    double r8 = k[7] * y9 * y6;
    double r9 = k[8] * y11 * y2;
    double r10 = k[9] * y11 * y1;
    double r11 = k[10] * y13;
    double r12 = k[11] * y10 * y2;
    double r13 = k[12] * y14;
    double r14 = k[13] * y1 * y6;
    double r15 = k[14] * y3;
    double r16 = k[15] * y4;
    double r17 = k[16] * y4;
    double r18 = k[17] * y16;
    double r19 = k[18] * y16;
    double r20 = k[19] * y17 * y6;
    double r21 = k[20] * y19;
    double r22 = k[21] * y19;
    double r23 = k[22] * y1 * y4;
    double r24 = k[23] * y19 * y1;
    double r25 = k[24] * y20;

    dydx[0] = -r1 - r10 - r14 - r23 - r24 + r2 + r3 + r9 + r11 + r12 + r22 + r25;
    dydx[1] = -r2 - r3 - r9 - r12 + r1 + r21;
    dydx[2] = -r15 + r1 + r17 + r19 + r22;
    dydx[3] = -r2 - r16 - r17 - r23 + r15;
    dydx[4] = -r3 + 2*r4 + r6 + r7 + r13 + r20;
    dydx[5] = -r6 - r8 - r14 - r20 + r3 + 2*r18;
    dydx[6] = -r4 - r5 - r6 + r13;
    dydx[7] = r4 + r5 + r6 + r7;
    dydx[8] = -r7 - r8;
    dydx[9] = -r12 + r7 + r9;
    dydx[10] = -r9 - r10 + r8 + r11;
    dydx[11] = r9;
    dydx[12] = -r11 + r10;
    dydx[13] = -r13 + r12;
    dydx[14] = r14;
    dydx[15] = -r18 - r19 + r16;
    dydx[16] = -r20;
    dydx[17] = r20;
    dydx[18] = -r21 - r22 - r24 + r23 + r25;
    dydx[19] = -r25 + r24;
}

void PollutionSystem::jacobian
(
    const double x,
    const std::vector<double>& y,
    std::vector<double>& dfdx,
    std::vector<std::vector<double>>& dfdy
) const
{
    size_t n = nEqns();
    
    for (size_t i = 0; i < n; ++i)
    {
        dfdx[i] = 0.0;
    }
    
    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j < n; ++j)
        {
            dfdy[i][j] = 0.0;
        }
    }
    
    double y1 = y[0];
    double y2 = y[1];
    double y3 = y[2];
    double y4 = y[3];
    double y5 = y[4];
    double y6 = y[5];
    double y7 = y[6];
    double y9 = y[8];
    double y10 = y[9];
    double y11 = y[10];
    double y13 = y[12];
    double y14 = y[13];
    double y16 = y[15];
    double y17 = y[16];
    double y19 = y[18];
    double y20 = y[19];

    dfdy[0][0] = -k[0] - k[9]*y11 - k[13]*y6 - k[22]*y4 - k[23]*y19;
    dfdy[0][1] = k[1]*y4 + k[2]*y5 + k[8]*y11 + k[11]*y10;
    dfdy[0][3] = k[1]*y2 - k[22]*y1;
    dfdy[0][4] = k[2]*y2;
    dfdy[0][5] = -k[13]*y1;
    dfdy[0][10] = -k[9]*y1 + k[8]*y2;
    dfdy[0][12] = k[10];
    dfdy[0][9] = k[11]*y2;
    dfdy[0][18] = k[21] - k[23]*y1;
    dfdy[0][19] = k[24];

    dfdy[1][0] = k[0];
    dfdy[1][1] = -k[1]*y4 - k[2]*y5 - k[8]*y11 - k[11]*y10;
    dfdy[1][3] = -k[1]*y2;
    dfdy[1][4] = -k[2]*y2;
    dfdy[1][10] = -k[8]*y2;
    dfdy[1][9] = -k[11]*y2;
    dfdy[1][18] = k[20];

    dfdy[2][0] = k[0];
    dfdy[2][2] = -k[14];
    dfdy[2][3] = k[16];
    dfdy[2][15] = k[18];
    dfdy[2][18] = k[21];

    dfdy[3][1] = -k[1]*y4;
    dfdy[3][2] = k[14];
    dfdy[3][3] = -k[1]*y2 - k[15] - k[16] - k[22]*y1;
    dfdy[3][0] = -k[22]*y4;

    dfdy[4][1] = -k[2]*y5;
    dfdy[4][4] = -k[2]*y2;
    dfdy[4][6] = 2*k[3] + k[5]*y6 + k[6];
    dfdy[4][5] = k[5]*y7;
    dfdy[4][8] = k[6];
    dfdy[4][13] = k[12];
    dfdy[4][16] = k[19]*y6;

    dfdy[5][1] = k[2]*y5;
    dfdy[5][4] = k[2]*y2;
    dfdy[5][5] = -k[5]*y7 - k[7]*y9 - k[13]*y1 - k[19]*y17;
    dfdy[5][6] = -k[5]*y6;
    dfdy[5][8] = -k[7]*y6;
    dfdy[5][0] = -k[13]*y6;
    dfdy[5][15] = 2*k[17];
    dfdy[5][16] = -k[19]*y6;

    dfdy[6][4] = -k[12];
    dfdy[6][6] = -k[3] - k[4] - k[5]*y6;
    dfdy[6][5] = -k[5]*y7;
    dfdy[6][13] = k[12];

    dfdy[7][6] = k[3] + k[4] + k[5]*y6;
    dfdy[7][5] = k[5]*y7;
    dfdy[7][8] = k[6];

    dfdy[8][6] = -k[6];
    dfdy[8][8] = -k[6] - k[7]*y6;
    dfdy[8][5] = -k[7]*y9;

    dfdy[9][9] = -k[11]*y2;
    dfdy[9][8] = k[6];
    dfdy[9][1] = -k[11]*y10 + k[8]*y11;
    dfdy[9][10] = k[8]*y2;

    dfdy[10][10] = -k[8]*y2 - k[9]*y1;
    dfdy[10][1] = -k[8]*y11;
    dfdy[10][0] = -k[9]*y11;
    dfdy[10][8] = k[7]*y6;
    dfdy[10][5] = k[7]*y9;
    dfdy[10][12] = k[10];

    dfdy[11][1] = k[8]*y11;
    dfdy[11][10] = k[8]*y2;

    dfdy[12][0] = k[9]*y11;
    dfdy[12][10] = k[9]*y1;
    dfdy[12][12] = -k[10];

    dfdy[13][9] = k[11]*y2;
    dfdy[13][1] = k[11]*y10;
    dfdy[13][13] = -k[12];

    dfdy[14][0] = k[13]*y6;
    dfdy[14][5] = k[13]*y1;

    dfdy[15][3] = k[15];
    dfdy[15][15] = -k[17] - k[18];

    dfdy[16][5] = -k[19]*y17;
    dfdy[16][16] = -k[19]*y6;

    dfdy[17][5] = k[19]*y17;
    dfdy[17][16] = k[19]*y6;

    dfdy[18][0] = -k[23]*y19 + k[22]*y4;
    dfdy[18][3] = k[22]*y1;
    dfdy[18][18] = -k[20] - k[21] - k[23]*y1;
    dfdy[18][19] = k[24];

    dfdy[19][0] = k[23]*y19;
    dfdy[19][18] = k[23]*y1;
    dfdy[19][19] = -k[24];
}

std::vector<double> PollutionSystem::getInitialConditions()
{
    std::vector<double> y0(20, 0.0);
    y0[0] = 0.0;
    y0[1] = 0.2;
    y0[2] = 0.0;
    y0[3] = 0.04;
    y0[4] = 0.0;
    y0[5] = 0.0;
    y0[6] = 0.1;
    y0[7] = 0.3;
    y0[8] = 0.01;
    y0[9] = 0.0;
    y0[10] = 0.0;
    y0[11] = 0.0;
    y0[12] = 0.0;
    y0[13] = 0.0;
    y0[14] = 0.0;
    y0[15] = 0.0;
    y0[16] = 0.007;
    y0[17] = 0.0;
    y0[18] = 0.0;
    y0[19] = 0.0;
    return y0;
}

std::vector<double> PollutionSystem::getGroundSolution()
{
    std::vector<double> yGround(20, 0.0);
    yGround[0] = 0.5646255480022769e-1;
    yGround[1] = 0.1342484130422339;
    yGround[2] = 0.4139734331099427e-8;
    yGround[3] = 0.5523140207484359e-2;
    yGround[4] = 0.2018977262302196e-6;
    yGround[5] = 0.1464541863493966e-6;
    yGround[6] = 0.7784249118997964e-1;
    yGround[7] = 0.3245075353396018;
    yGround[8] = 0.7494013383880406e-2;
    yGround[9] = 0.1622293157301561e-7;
    yGround[10] = 0.1135863833257075e-7;
    yGround[11] = 0.2230505975721359e-2;
    yGround[12] = 0.2087162882798630e-3;
    yGround[13] = 0.1396921016840158e-4;
    yGround[14] = 0.8964884856898295e-2;
    yGround[15] = 0.4352846369330103e-17;
    yGround[16] = 0.6899219696263405e-2;
    yGround[17] = 0.1007803037365946e-3;
    yGround[18] = 0.1772146513969984e-5;
    yGround[19] = 0.5682943292316392e-4;
    return yGround;
}

} // namespace kodes