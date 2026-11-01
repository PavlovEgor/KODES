#ifndef KODES_ROSENBROCK_TABLEAUX
#define KODES_ROSENBROCK_TABLEAUX

#pragma once

#include "basic_types.cuh"

namespace kodes
{

// The coefficients of the Rosenbrock methods, all in the one form Rosenbrock
// takes them in:
//
//     (I/(kGamma*dt) - df/dy) k_i = f(y0 + sum_j kA[i][j] k_j) + sum_j kC[i][j]/dt k_j
//
//     y1  = y0 + sum_i kM[i] k_i
//     err =      sum_i kE[i] k_i
//
// kNewF[i] false: stage i is taken at the state of stage i-1, kA repeats the
// row above, and the derivatives are not evaluated again.
//
// The values are OpenFOAM's, which leaves out the df/dt terms d_i as well,
// since the systems here are autonomous. Each was checked by its measured
// order of the step and of the error estimate, on a nonlinear system, and by
// its stability function far down the negative real axis.

// Rosenbrock23: order 3, error estimate of order 2, 3 stages, L-stable.
// Sandu et al.'s ROS3.
struct Rosenbrock23Tableau
{
    static constexpr label kStages = 3;

    static constexpr scalar kGamma = 0.43586652150845899941601945119356;

    static constexpr scalar kA[kStages][kStages] =
    {
        {0, 0, 0},
        {1, 0, 0},
        {1, 0, 0}
    };

    static constexpr scalar kC[kStages][kStages] =
    {
        {0, 0, 0},
        {-0.10156171083877702091975600115545e+01, 0, 0},
        {0.40759956452537699824805835358067e+01, 0.92076794298330791242156818474003e+01, 0}
    };

    static constexpr scalar kM[kStages] =
    {
        1, 0.61697947043828245592553615689730e+01, -0.42772256543218573326238373806514
    };

    static constexpr scalar kE[kStages] =
    {
        0.5, -0.29079558716805469821718236208017e+01, 0.22354069897811569627360909276199
    };

    static constexpr bool kNewF[kStages] = {true, true, false};
};

// Rosenbrock34: order 4, error estimate of order 3, 4 stages, L-stable. The
// L-stable coefficients of Hairer and Wanner, OpenFOAM's choice over
// Shampine's, which are more accurate for small steps but only A-stable.
struct Rosenbrock34Tableau
{
    static constexpr label kStages = 4;

    static constexpr scalar kGamma = 0.57282;

    static constexpr scalar kA[kStages][kStages] =
    {
        {0, 0, 0, 0},
        {2, 0, 0, 0},
        {1.867943637803922, 0.2344449711399156, 0, 0},
        {1.867943637803922, 0.2344449711399156, 0, 0}
    };

    static constexpr scalar kC[kStages][kStages] =
    {
        {0, 0, 0, 0},
        {-7.137615036412310, 0, 0, 0},
        {2.580708087951457, 0.6515950076447975, 0, 0},
        {-2.137148994382534, -0.3214669691237626, -0.6949742501781779, 0}
    };

    static constexpr scalar kM[kStages] =
    {
        2.255570073418735, 0.2870493262186792, 0.435317943184018, 1.093502252409163
    };

    static constexpr scalar kE[kStages] =
    {
        -0.2815431932141155, -0.0727619912493892, -0.1082196201495311, -1.093502252409163
    };

    static constexpr bool kNewF[kStages] = {true, true, true, false};
};

// rodas23: order 3, error estimate of order 2, 4 stages, L-stable and stiffly
// accurate - the last stage is the solution. Sandu et al.'s RODAS3.
struct Rodas23Tableau
{
    static constexpr label kStages = 4;

    static constexpr scalar kGamma = 0.5;

    static constexpr scalar kA[kStages][kStages] =
    {
        {0, 0, 0, 0},
        {0, 0, 0, 0},
        {2, 0, 0, 0},
        {2, 0, 1, 0}
    };

    static constexpr scalar kC[kStages][kStages] =
    {
        {0, 0, 0, 0},
        {4, 0, 0, 0},
        {1, -1, 0, 0},
        {1, -1, -8.0/3.0, 0}
    };

    static constexpr scalar kM[kStages] = {2, 0, 1, 1};

    static constexpr scalar kE[kStages] = {0, 0, 0, 1};

    static constexpr bool kNewF[kStages] = {true, false, true, true};
};

// rodas34: order 4, error estimate of order 3, 6 stages, L-stable and stiffly
// accurate. Hairer and Wanner's RODAS.
struct Rodas34Tableau
{
    static constexpr label kStages = 6;

    static constexpr scalar kGamma = 0.25;

    static constexpr scalar kA[kStages][kStages] =
    {
        {0, 0, 0, 0, 0, 0},
        {0.1544e1, 0, 0, 0, 0, 0},
        {0.9466785280815826, 0.2557011698983284, 0, 0, 0, 0},
        {0.3314825187068521e1, 0.2896124015972201e1, 0.9986419139977817, 0, 0, 0},
        {0.1221224509226641e1, 0.6019134481288629e1, 0.1253708332932087e2, -0.6878860361058950, 0, 0},
        {0.1221224509226641e1, 0.6019134481288629e1, 0.1253708332932087e2, -0.6878860361058950, 1, 0}
    };

    static constexpr scalar kC[kStages][kStages] =
    {
        {0, 0, 0, 0, 0, 0},
        {-0.56688e1, 0, 0, 0, 0, 0},
        {-0.2430093356833875e1, -0.2063599157091915, 0, 0, 0, 0},
        {-0.1073529058151375, -0.9594562251023355e1, -0.2047028614809616e2, 0, 0, 0},
        {0.7496443313967647e1, -0.1024680431464352e2, -0.3399990352819905e2, 0.1170890893206160e2, 0, 0},
        {0.8083246795921522e1, -0.7981132988064893e1, -0.3152159432874371e2, 0.1631930543123136e2, -0.6058818238834054e1, 0}
    };

    static constexpr scalar kM[kStages] =
    {
        0.1221224509226641e1, 0.6019134481288629e1, 0.1253708332932087e2, -0.6878860361058950, 1, 1
    };

    static constexpr scalar kE[kStages] = {0, 0, 0, 0, 0, 1};

    static constexpr bool kNewF[kStages] = {true, true, true, true, true, true};
};

}

#endif
