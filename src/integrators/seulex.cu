#include "seulex.cuh"
#include <algorithm>

#define SMALL 1e-9
#define GREAT 1e9
#define sqr(x) ((x)*(x))
#define min(a, b) (((a) > (b)) ? (b) : (a) )
#define max(a, b) (((a) > (b)) ? (a) : (b) )

void LUDecompose
(
    std::vector<std::vector<double>>& matrix,
    //! [out] size is adjusted as required
    std::vector<int>& pivotIndices
);

void LUDecompose
(
    std::vector<std::vector<double>>& matrix,
    //! [out] size is adjusted as required
    std::vector<int>& pivotIndices,
    //! [out] is -1 for odd number of row interchanges and 1 for even number
    int& sign
);

template<class Type>
void LUBacksubstitute
(
    const std::vector<std::vector<double>>& luMatrix,
    const std::vector<int> pivotIndices,
    std::vector<Type>& source
);

namespace kodes 
{
const double
        seulex::stepFactor1_ = 0.6,
        seulex::stepFactor2_ = 0.93,
        seulex::stepFactor3_ = 0.1,
        seulex::stepFactor4_ = 4,
        seulex::stepFactor5_ = 0.5,
        seulex::kFactor1_ = 0.7,
        seulex::kFactor2_ = 0.9;
}

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

kodes::seulex::seulex(const ODESystem& ode, const kodes::Config& config)
:
    integrator(ode, config),
    jacRedo_(min(1e-4, static_cast<double>(*std::min_element(relTol_.begin(), relTol_.end())))),
    nSeq_(iMaxx_),
    cpu_(iMaxx_),
    coeff_(iMaxx_, std::vector<double>(iMaxx_, 0.0)),
    theta_(2*jacRedo_),
    table_(kMaxx_, std::vector<double>(sizeOfSystem_, 0.0)),
    dfdx_(sizeOfSystem_),
    dfdy_(sizeOfSystem_),
    a_(sizeOfSystem_),
    pivotIndices_(sizeOfSystem_),
    dxOpt_(iMaxx_),
    temp_(iMaxx_),
    y0_(sizeOfSystem_),
    ySequence_(sizeOfSystem_),
    scale_(sizeOfSystem_),
    dy_(sizeOfSystem_),
    yTemp_(sizeOfSystem_),
    dydx_(sizeOfSystem_)
{
    // The CPU time factors for the major parts of the algorithm
    const double cpuFunc = 1, cpuJac = 5, cpuLU = 1, cpuSolve = 1;

    nSeq_[0] = 2;
    nSeq_[1] = 3;

    for (int i=2; i<iMaxx_; i++)
    {
        nSeq_[i] = 2*nSeq_[i-2];
    }
    cpu_[0] = cpuJac + cpuLU + nSeq_[0]*(cpuFunc + cpuSolve);

    for (int k=0; k<kMaxx_; k++)
    {
        cpu_[k+1] = cpu_[k] + (nSeq_[k+1]-1)*(cpuFunc + cpuSolve) + cpuLU;
    }

    // Set the extrapolation coefficients array
    for (int k=0; k<iMaxx_; k++)
    {
        for (int l=0; l<k; l++)
        {
            double ratio = double(nSeq_[k])/nSeq_[l];
            coeff_[k][l] = 1/(ratio - 1);
        }
    }
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

bool kodes::seulex::seul
(
    const double x0,
    const std::vector<double>& y0,
    const double dxTot,
    const size_t k,
    std::vector<double>& y,
    const std::vector<double>& scale
) const
{
    size_t nSteps = nSeq_[k];
    double dx = dxTot/nSteps;

    for (size_t i=0; i<sizeOfSystem_; i++)
    {
        for (size_t j=0; j<sizeOfSystem_; j++)
        {
            a_[i][j] = -dfdy_[i][j];
        }

        a_[i][i] += 1/dx;
    }

    LUDecompose(a_, pivotIndices_);

    double xnew = x0 + dx;
    odes_.derivatives(xnew, y0, dy_);
    LUBacksubstitute(a_, pivotIndices_, dy_);

    yTemp_ = y0;

    for (size_t nn=1; nn<nSteps; nn++)
    {
        for(size_t i=0; i<sizeOfSystem_; ++i)
        {
            yTemp_[i] += dy_[i];
        }

        xnew += dx;

        if (nn == 1 && k<=1)
        {
            double dy1 = 0;
            for (size_t i=0; i<sizeOfSystem_; i++)
            {
                dy1 += sqr(dy_[i]/scale[i]);
            }
            dy1 = sqrt(dy1);

            odes_.derivatives(x0 + dx, yTemp_, dydx_);
            for (size_t i=0; i<sizeOfSystem_; i++)
            {
                dy_[i] = dydx_[i] - dy_[i]/dx;
            }

            LUBacksubstitute(a_, pivotIndices_, dy_);

            const double denom = min(1, dy1 + SMALL);
            double dy2 = 0;
            for (size_t i=0; i<sizeOfSystem_; i++)
            {
                // Test of dy_[i] to avoid overflow
                if (fabs(dy_[i]) > scale[i]*denom)
                {
                    theta_ = 1;
                    return false;
                }

                dy2 += sqr(dy_[i]/scale[i]);
            }
            dy2 = sqrt(dy2);
            theta_ = dy2/denom;

            if (theta_ > 1)
            {
                return false;
            }
        }

        odes_.derivatives(xnew, yTemp_, dy_);
        LUBacksubstitute(a_, pivotIndices_, dy_);
    }

    for (size_t i=0; i<sizeOfSystem_; i++)
    {
        y[i] = yTemp_[i] + dy_[i];
    }

    return true;
}


void kodes::seulex::extrapolate
(
    const size_t k,
    std::vector<std::vector<double>>& table,
    std::vector<double>& y
) const
{
    for (int j=k-1; j>0; j--)
    {
        for (size_t i=0; i<sizeOfSystem_; i++)
        {
            table[j-1][i] =
                table[j][i] + coeff_[k][j]*(table[j][i] - table[j-1][i]);
        }
    }

    for (int i=0; i<sizeOfSystem_; i++)
    {
        y[i] = table[0][i] + coeff_[k][0]*(table[0][i] - y[i]);
    }
}


void kodes::seulex::solve
(
    double& x,
    std::vector<double>& y,
    stepState& step
) const
{
    temp_[0] = GREAT;
    double dx = step.dxTry;
    y0_ = y;
    dxOpt_[0] = fabs(0.1*dx);

    if (step.first || step.prevReject)
    {
        theta_ = 2*jacRedo_;
    }

    if (step.first)
    {
        // NOTE: the first element of relTol_ and absTol_ are used here.
        double logTol = -log10(relTol_[0] + absTol_[0])*0.6 + 0.5;
        kTarg_ = max(1, min(kMaxx_ - 1, int(logTol)));
    }

    for (size_t i=0; i < sizeOfSystem_; ++i)
    {
        scale_[i] = absTol_[i] + relTol_[i]*fabs(y[i]);
    }

    bool jacUpdated = false;

    if (theta_ > jacRedo_)
    {
        odes_.jacobian(x, y, dfdx_, dfdy_);
        jacUpdated = true;
    }

    int k;
    double dxNew = fabs(dx);
    bool firstk = true;

    while (firstk || step.reject)
    {
        dx = step.forward ? dxNew : -dxNew;
        firstk = false;
        step.reject = false;

        // if (fabs(dx) <= fabs(x)*sqr(SMALL))
        // {
        //     std::
        //             << "step size underflow :"  << dx << endl;
        // }

        double errOld = 0;

        for (k=0; k<=kTarg_+1; k++)
        {
            bool success = seul(x, y0_, dx, k, ySequence_, scale_);

            if (!success)
            {
                step.reject = true;
                dxNew = fabs(dx)*stepFactor5_;
                break;
            }

            if (k == 0)
            {
                 y = ySequence_;
            }
            else
            {
                for (size_t i=0; i<sizeOfSystem_; ++i)
                {
                    table_[k-1][i] = ySequence_[i];
                }
            }

            if (k != 0)
            {
                extrapolate(k, table_, y);
                double err = 0;
                for (size_t i=0; i<sizeOfSystem_; ++i)
                {
                    scale_[i] = absTol_[i] + relTol_[i]*fabs(y0_[i]);
                    err += sqr((y[i] - table_[0][i])/scale_[i]);
                }
                err = sqrt(err/sizeOfSystem_);
                if (err > 1/SMALL || (k > 1 && err >= errOld))
                {
                    step.reject = true;
                    dxNew = fabs(dx)*stepFactor5_;
                    break;
                }
                errOld = min(4*err, 1);
                double expo = 1.0/(k + 1);
                double facmin = pow(stepFactor3_, expo);
                double fac;
                if (err == 0)
                {
                    fac = 1/facmin;
                }
                else
                {
                    fac = stepFactor2_/pow(err/stepFactor1_, expo);
                    fac = max(facmin/stepFactor4_, min(1/facmin, fac));
                }
                dxOpt_[k] = fabs(dx*fac);
                temp_[k] = cpu_[k]/dxOpt_[k];

                if ((step.first || step.last) && err <= 1)
                {
                    break;
                }

                if
                (
                    k == kTarg_ - 1
                 && !step.prevReject
                 && !step.first && !step.last
                )
                {
                    if (err <= 1)
                    {
                        break;
                    }
                    else if (err > nSeq_[kTarg_]*nSeq_[kTarg_ + 1]*4)
                    {
                        step.reject = true;
                        kTarg_ = k;
                        if (kTarg_>1 && temp_[k-1] < kFactor1_*temp_[k])
                        {
                            kTarg_--;
                        }
                        dxNew = dxOpt_[kTarg_];
                        break;
                    }
                }

                if (k == kTarg_)
                {
                    if (err <= 1)
                    {
                        break;
                    }
                    else if (err > nSeq_[k + 1]*2)
                    {
                        step.reject = true;
                        if (kTarg_>1 && temp_[k-1] < kFactor1_*temp_[k])
                        {
                            kTarg_--;
                        }
                        dxNew = dxOpt_[kTarg_];
                        break;
                    }
                }

                if (k == kTarg_+1)
                {
                    if (err > 1)
                    {
                        step.reject = true;
                        if
                        (
                            kTarg_ > 1
                         && temp_[kTarg_-1] < kFactor1_*temp_[kTarg_]
                        )
                        {
                            kTarg_--;
                        }
                        dxNew = dxOpt_[kTarg_];
                    }
                    break;
                }
            }
        }
        if (step.reject)
        {
            step.prevReject = true;
            if (!jacUpdated)
            {
                theta_ = 2*jacRedo_;

                if (theta_ > jacRedo_ && !jacUpdated)
                {
                    odes_.jacobian(x, y, dfdx_, dfdy_);
                    jacUpdated = true;
                }
            }
        }
    }

    jacUpdated = false;

    step.dxDid = dx;
    x += dx;

    size_t kopt;
    if (k == 1)
    {
        kopt = 2;
    }
    else if (k <= kTarg_)
    {
        kopt=k;
        if (temp_[k-1] < kFactor1_*temp_[k])
        {
            kopt = k - 1;
        }
        else if (temp_[k] < kFactor2_*temp_[k - 1])
        {
            kopt = min(k + 1, kMaxx_ - 1);
        }
    }
    else
    {
        kopt = k - 1;
        if (k > 2 && temp_[k-2] < kFactor1_*temp_[k - 1])
        {
            kopt = k - 2;
        }
        if (temp_[k] < kFactor2_*temp_[kopt])
        {
            kopt = min(k, kMaxx_ - 1);
        }
    }

    if (step.prevReject)
    {
        kTarg_ = min(kopt, k);
        dxNew = min(fabs(dx), dxOpt_[kTarg_]);
        step.prevReject = false;
    }
    else
    {
        if (kopt <= k)
        {
            dxNew = dxOpt_[kopt];
        }
        else
        {
            if (k < kTarg_ && temp_[k] < kFactor2_*temp_[k - 1])
            {
                dxNew = dxOpt_[k]*cpu_[kopt + 1]/cpu_[k];
            }
            else
            {
                dxNew = dxOpt_[k]*cpu_[kopt]/cpu_[k];
            }
        }
        kTarg_ = kopt;
    }

    step.dxTry = step.forward ? dxNew : -dxNew;
}


// ************************************************************************* //

void LUDecompose
(
    std::vector<std::vector<double>>& matrix,
    std::vector<int>& pivotIndices
)
{
    int sign;
    LUDecompose(matrix, pivotIndices, sign);
}

void LUDecompose
(
    std::vector<std::vector<double>>& matrix,
    std::vector<int>& pivotIndices,
    int& sign
)
{
    const size_t size = matrix.size();
    std::vector<double> vv(size);
    sign = 1;

    for (size_t i = 0; i < size; ++i)
    {
        double largestCoeff = 0.0;
        double temp;
        const double* __restrict__ matrixi = matrix[i].data();

        for (size_t j = 0; j < size; ++j)
        {
            if ((temp = fabs(matrixi[j])) > largestCoeff)
            {
                largestCoeff = temp;
            }
        }

        // if (largestCoeff == 0.0)
        // {
        //     FatalErrorInFunction
        //         << "Singular matrix" << exit(FatalError);
        // }

        vv[i] = 1.0/largestCoeff;
    }

    for (size_t j = 0; j < size; ++j)
    {
        double* __restrict__ matrixj = matrix[j].data();

        for (size_t i = 0; i < j; ++i)
        {
            double* __restrict__ matrixi = matrix[i].data();

            double sum = matrixi[j];
            for (size_t k = 0; k < i; ++k)
            {
                sum -= matrixi[k]*matrix[k][j];
            }
            matrixi[j] = sum;
        }

        size_t iMax = 0;

        double largestCoeff = 0.0;
        for (size_t i = j; i < size; ++i)
        {
            double* __restrict__ matrixi = matrix[i].data();
            double sum = matrixi[j];

            for (size_t k = 0; k < j; ++k)
            {
                sum -= matrixi[k]*matrix[k][j];
            }

            matrixi[j] = sum;

            double temp;
            if ((temp = vv[i]*fabs(sum)) >= largestCoeff)
            {
                largestCoeff = temp;
                iMax = i;
            }
        }

        pivotIndices[j] = iMax;

        if (j != iMax)
        {
            double* __restrict__ matrixiMax = matrix[iMax].data();

            for (size_t k = 0; k < size; ++k)
            {
                std::swap(matrixj[k], matrixiMax[k]);
            }

            sign *= -1;
            vv[iMax] = vv[j];
        }

        if (matrixj[j] == 0.0)
        {
            matrixj[j] = SMALL;
        }

        if (j != size-1)
        {
            double rDiag = 1.0/matrixj[j];

            for (size_t i = j + 1; i < size; ++i)
            {
                matrix[i][j] *= rDiag;
            }
        }
    }
}

template<class Type>
void LUBacksubstitute
(
    const std::vector<std::vector<double>>& luMatrix,
    const std::vector<int> pivotIndices,
    std::vector<Type>& sourceSol
)
{
    size_t m = luMatrix.size();

    size_t ii = 0;

    for (size_t i = 0; i < m; ++i)
    {
        size_t ip = pivotIndices[i];
        Type sum = sourceSol[ip];
        sourceSol[ip] = sourceSol[i];
        const double* __restrict__ luMatrixi = luMatrix[i].data();

        if (ii != 0)
        {
            for (size_t j = ii - 1; j < i; ++j)
            {
                sum -= luMatrixi[j]*sourceSol[j];
            }
        }
        else if (sum != Type(0))
        {
            ii = i + 1;
        }

        sourceSol[i] = sum;
    }

    for (size_t i = m - 1; i >= 0; --i)
    {
        Type sum = sourceSol[i];
        const double* __restrict__ luMatrixi = luMatrix[i].data();

        for (size_t j = i + 1; j < m; ++j)
        {
            sum -= luMatrixi[j]*sourceSol[j];
        }

        sourceSol[i] = sum/luMatrixi[i];
    }
}