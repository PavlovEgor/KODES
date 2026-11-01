#include <cfloat>

#include "Bdf.cuh"
#include "basic_linalg.cuh"
#include "thread_vector_functions.cuh"

KODES_DEFINE_DEVICE_OBJECT(kodes::Bdf)

__host__ void
kodes::Bdf::allocate()
{
    IntegrationMethod::allocate();

    table_.allocate(scratchSize_);
    dfdy_.allocate(scratchSize_);
    a_.allocate(scratchSize_);

    pivotIndices_.allocate(scratchSize_);
    vv_.allocate(scratchSize_);
    ru_.allocate(scratchSize_);

    yPredict_.allocate(scratchSize_);
    psi_.allocate(scratchSize_);
    scale_.allocate(scratchSize_);
    d_.allocate(scratchSize_);
    dy_.allocate(scratchSize_);

    order_.allocate(scratchSize_);
    nEqualSteps_.allocate(scratchSize_);
    hAbs_.allocate(scratchSize_);
    luValid_.allocate(scratchSize_);
}

__host__ void
kodes::Bdf::deallocate()
{
    table_.deallocate();
    dfdy_.deallocate();
    a_.deallocate();

    pivotIndices_.deallocate();
    vv_.deallocate();
    ru_.deallocate();

    yPredict_.deallocate();
    psi_.deallocate();
    scale_.deallocate();
    d_.deallocate();
    dy_.deallocate();

    order_.deallocate();
    nEqualSteps_.deallocate();
    hAbs_.deallocate();
    luValid_.deallocate();

    IntegrationMethod::deallocate();
}

__device__ void
kodes::Bdf::startSystem(kodes::ODESystem* ode) const
{
    const ThreadVector<scalar> y = currentVector();
    const DifferenceTable table = table_;
    const scalar dt = deltaTTry;

    table.column(0) = y;

    ode->derivatives(currentT, currentParameter(0), y, dy_);
    table.column(1) = dt*dy_;

    order_ = 1;
    nEqualSteps_ = 0;
    hAbs_ = fabs(dt);
    luValid_ = false;

    ode->jacobian(currentT, currentParameter(0), y, dfdy_);
}

__device__ void
kodes::Bdf::rescale(const label order, const scalar factor) const
{
    // a copy, so that the loops do not read the buffer back out of this object
    const DifferenceTable table = table_;

    table.rescale(order, factor, ru_);

    nEqualSteps_ = 0;
}

__device__ bool
kodes::Bdf::newton
(
    kodes::ODESystem* ode,
    const scalar tNew,
    const scalar c,
    const scalar newtonTol,
    label& nIter
) const
{
    const ThreadVector<scalar> y = currentVector();

    d_ = 0.0;
    y = yPredict_;

    scalar dyNormOld = 0;

    for (label k=0; k<kBdfNewtonMaxIter; k++)
    {
        nIter = k + 1;

        ode->derivatives(tNew, currentParameter(0), y, dy_);

        bool finite = true;

        {
            const ThreadVector<scalar> dy = dy_;
            const ThreadVector<const scalar> psi = psi_;
            const ThreadVector<const scalar> d = d_;

            #pragma unroll 4
            for (label i=0; i<kSystemSize; i++)
            {
                finite = finite && isfinite(dy[i]);
                dy[i] = c*dy[i] - psi[i] - d[i];
            }
        }

        if (!finite)
        {
            return false;
        }

        LUBacksubstitute(a_, pivotIndices_, dy_);

        const scalar dyNorm = rms(dy_/scale_);

        // The rate of convergence, from the second iteration on: diverging,
        // or too slow to get within tolerance in the iterations left
        scalar rate = 0;

        if (k > 0)
        {
            rate = dyNorm/dyNormOld;

            if
            (
                rate >= 1
             || pow(rate, scalar(kBdfNewtonMaxIter - k))/(1 - rate)*dyNorm > newtonTol
            )
            {
                return false;
            }
        }

        y += dy_;
        d_ += dy_;

        if (dyNorm == 0 || (k > 0 && rate/(1 - rate)*dyNorm < newtonTol))
        {
            return true;
        }

        dyNormOld = dyNorm;
    }

    return false;
}

__device__ void
kodes::Bdf::chooseOrderAndStep
(
    const label order,
    const scalar errorNorm,
    const scalar safety
) const
{
    const DifferenceTable table = table_;

    // The error orders order-1 and order+1 would have made, infinite where
    // there is no such order: a factor of 0 below
    const scalar errorM =
        order > 1
      ? rms(kBdfCoefficients.errorConst[order - 1]*table.column(order)/scale_)
      : INFINITY;

    const scalar errorP =
        order < kBdfMaxOrder
      ? rms(kBdfCoefficients.errorConst[order + 1]*table.column(order + 2)/scale_)
      : INFINITY;

    // The step factor each order allows; the first of the largest wins, as
    // np.argmax picks it - a tie with the lower order goes to the lower order
    const scalar factorM = pow(errorM, -1.0/order);
    const scalar factor0 = pow(errorNorm, -1.0/(order + 1));
    const scalar factorP = pow(errorP, -1.0/(order + 2));

    label newOrder = order;
    scalar factor = factor0;

    if (factorM >= factor)
    {
        newOrder = order - 1;
        factor = factorM;
    }

    if (factorP > factor)
    {
        newOrder = order + 1;
        factor = factorP;
    }

    factor = min(kBdfMaxFactor, safety*factor);

    order_ = newOrder;
    hAbs_ = hAbs_*factor;
    rescale(newOrder, factor);
    luValid_ = false;
}

// One accepted step: scipy's BDF._step_impl, with what was its state in
// scratch space. What scipy does for a step clipped to the end of the
// interval - rescale the history to it - is done here for any step the
// integrator hands in that is not the one this method asked for.
__device__ void
kodes::Bdf::advance
(
    kodes::ODESystem* ode,
    kodes::IntegratorControls controls
) const
{
    const scalar atol = controls.absTol;
    const scalar rtol = controls.relTol;
    const scalar newtonTol = max(10*DBL_EPSILON/rtol, min(0.03, sqrt(rtol)));

    scalar& t = currentT;

    if (first)
    {
        startSystem(ode);
    }
    else if (const scalar dtAbs = fabs(scalar(deltaTTry)); dtAbs != hAbs_)
    {
        rescale(order_, dtAbs/hAbs_);
        hAbs_ = dtAbs;
        luValid_ = false;
    }

    const label order = order_;
    const scalar alpha = kBdfCoefficients.alpha[order];
    const scalar minStep = 10*fabs(nextafter(t, forward ? INFINITY : -INFINITY) - t);

    scalar hAbs = hAbs_;
    bool jacobianCurrent = false;

    scalar dt;
    scalar errorNorm;
    scalar safety;

    for (;;)
    {
        if (hAbs < minStep)
        {
            printf("system %d: step size underflow : %0.16e \n", controls.system, hAbs);
        }

        dt = forward ? hAbs : -hAbs;
        const scalar tNew = t + dt;

        {
            const DifferenceTable table = table_;
            table.predict(order, yPredict_, psi_);
        }

        scale_ = atol + rtol*cmptMag(yPredict_);

        const scalar c = dt/alpha;

        // Factorise afresh only when c changed, or the Jacobian did; a
        // Jacobian is taken afresh only once a step, at the prediction, and
        // only when the iteration failed with the one it had
        bool converged = false;
        label nIter = 0;

        for (;;)
        {
            if (!luValid_)
            {
                const ThreadMatrix<scalar> a = a_;
                const ThreadMatrix<const scalar> dfdy = dfdy_;

                #pragma unroll 4
                for (label i=0; i<kSystemSize; i++)
                {
                    #pragma unroll 4
                    for (label j=0; j<kSystemSize; j++)
                    {
                        a(i, j) = -c*dfdy(i, j);
                    }
                    a(i, i) += 1;
                }

                LUDecompose(a_, pivotIndices_, vv_);
                luValid_ = true;
            }

            converged = newton(ode, tNew, c, newtonTol, nIter);

            if (converged || jacobianCurrent)
            {
                break;
            }

            ode->jacobian(tNew, currentParameter(0), yPredict_, dfdy_);
            luValid_ = false;
            jacobianCurrent = true;
        }

        if (!converged)
        {
            hAbs *= 0.5;
            rescale(order, 0.5);
            luValid_ = false;
            continue;
        }

        safety = 0.9*(2*kBdfNewtonMaxIter + 1)/(2*kBdfNewtonMaxIter + nIter);

        scale_ = atol + rtol*cmptMag(currentVector());
        errorNorm = rms(kBdfCoefficients.errorConst[order]*d_/scale_);

        if (errorNorm <= 1)
        {
            break;
        }

        // The iteration converged, so the factorisation is kept, though made
        // for the step before this one
        const scalar factor =
            max(kBdfMinFactor, safety*pow(errorNorm, -1.0/(order + 1)));

        hAbs *= factor;
        rescale(order, factor);
    }

    t += dt;
    deltaTDid = dt;
    hAbs_ = hAbs;

    {
        const DifferenceTable table = table_;
        table.update(order, d_);
    }

    nEqualSteps_ = nEqualSteps_ + 1;

    if (nEqualSteps_ >= order + 1)
    {
        chooseOrderAndStep(order, errorNorm, safety);
    }

    deltaTTry = forward ? hAbs_ : -hAbs_;
}
