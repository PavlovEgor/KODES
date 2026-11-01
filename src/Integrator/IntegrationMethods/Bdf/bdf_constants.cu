#include "bdf_constants.cuh"

namespace kodes
{

static constexpr BdfCoefficients bdfCoefficients()
{
    constexpr scalar kappa[kBdfTableSize] = {0, -0.1850, -1.0/9.0, -0.0823, -0.0415, 0};

    BdfCoefficients c{};

    c.gamma[0] = 0;

    for (label k = 1; k < kBdfTableSize; k++)
    {
        c.gamma[k] = c.gamma[k-1] + 1.0/k;
    }

    for (label k = 0; k < kBdfTableSize; k++)
    {
        c.alpha[k] = (1 - kappa[k])*c.gamma[k];
        c.errorConst[k] = kappa[k]*c.gamma[k] + 1.0/(k + 1);
    }

    for (label j = 0; j < kBdfTableSize; j++)
    {
        c.u[0][j] = 1;

        for (label m = 1; m < kBdfTableSize; m++)
        {
            c.u[m][j] = c.u[m-1][j]*(m - 1 - j)/m;
        }
    }

    return c;
}

__constant__ BdfCoefficients kBdfCoefficients = bdfCoefficients();

}
