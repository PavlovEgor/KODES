#include "Seulex.cuh"
#include "basic_linalg.cuh"
#include "block_reduce.cuh"

KODES_DEFINE_DEVICE_OBJECT(kodes::Seulex)

namespace kodes
{

// Which system of the block the shared Jacobian is evaluated at.
//
// The balancer sorts the batch by ascending key and a block takes blockDim.x
// consecutive positions of that order, so the block's *last* active lane holds
// its stiffest system. That is the one to linearise about: the step is
// linearly implicit, so a Jacobian stiffer than the system it is used on only
// over-damps the step and it is still accepted, while one too soft lets the
// contraction seul() measures go past 1 and the step is thrown away. Erring
// towards the stiff end therefore costs accuracy nothing and rejections much.
//
// Flip this if a balancer is ever added whose key falls as stiffness rises.
constexpr bool kJacobianAtStiffestLane = true;

}

__device__
bool kodes::Seulex::seul (
    kodes::SeulexDeviceResources* resources,
    const kodes::SeulexBlockScratch& block,
    kodes::ODESystem* ode,
    const label jacTid,
    const bool active,
    const scalar t0,
    const scalar dtTot,
    const label k,
    scalar& theta
)
{
    const scalar* __restrict__ dfdy = resources->dfdy();

    scalar* __restrict__ y0    = resources->y0();
    scalar* __restrict__ scale = resources->scale();

    scalar* __restrict__ dy    = resources->dy();
    scalar* __restrict__ yTemp = resources->yTemp();
    scalar* __restrict__ dydt  = resources->dydt();
    scalar* __restrict__ y     = resources->ySequence();

    const label systemSize = resources->systemSize();

    const label nSteps = kSeulexStepSequence[k];
    const scalar dt = dtTot/nSteps;

    // The block shares one matrix across every call of this function, and the
    // one below overwrites it. Without this barrier a thread that has raced
    // ahead to the next order would rebuild it while its neighbours were still
    // back-substituting through the factors of the last one - the step sizes
    // come out of a reduction, so they would agree on everything except the
    // answer.
    __syncthreads();

    // I/dt - J, built by the whole block out of the one Jacobian pyJac left in
    // the slot of thread jacTid. Flattened over the threads, so a 9x9 matrix
    // spreads over as many of them as a 53x53 one does.
    for (label t = threadIdx.x; t < systemSize*systemSize; t += blockDim.x)
    {
        const label i = t % systemSize;
        const label j = t / systemSize;

        block.a[INDEXBLOCKMAT(i, j, systemSize)] =
            (i == j ? 1/dt : 0.0)
          - dfdy[INDEXJACMAT(i, j, systemSize, jacTid)];
    }

    __syncthreads();

    blockLUDecompose(block.a, block.pivotIndices, block.vv, systemSize);

    scalar tNew = t0 + dt;

    if (active)
    {
        ode->derivatives(tNew, resources->currentParameter(0), y0, dy);

        LUBacksubstitute(block.a, block.pivotIndices, dy, systemSize);

        copyVec(yTemp, y0, systemSize);
    }

    for (label nn=1; nn<nSteps; nn++)
    {
        if (active)
        {
            sumVec(yTemp, yTemp, dy, systemSize);
        }

        tNew += dt;

        if (nn == 1 && k<=1)
        {
            scalar threadTheta = 0;
            bool overflow = false;

            if (active)
            {
                scalar dy1 = 0;
                for (label i=0; i<systemSize; i++)
                {
                    dy1 += sqr(dy[INDEXVEC(i)]/scale[INDEXVEC(i)]);
                }
                dy1 = sqrt(dy1);

                ode->derivatives(t0 + dt, resources->currentParameter(0), yTemp, dydt);
                for (label i=0; i<systemSize; i++)
                {
                    dy[INDEXVEC(i)] = dydt[INDEXVEC(i)] - dy[INDEXVEC(i)]/dt;
                }

                LUBacksubstitute(block.a, block.pivotIndices, dy, systemSize);

                const scalar denom = min(1.0, dy1 + SMALL);
                scalar dy2 = 0;
                for (label i=0; i<systemSize; i++)
                {
                    // Test of dy[i] to avoid overflow
                    if (fabs(dy[INDEXVEC(i)]) > scale[INDEXVEC(i)]*denom)
                    {
                        overflow = true;
                    }

                    dy2 += sqr(dy[INDEXVEC(i)]/scale[INDEXVEC(i)]);
                }
                dy2 = sqrt(dy2);
                threadTheta = dy2/denom;
            }

            // One system overflowing condemns the block's step: they all share
            // the matrix that produced it. Both exits below are taken by the
            // whole block, which is what lets them stand in front of the
            // barriers the rest of this function is made of.
            if (blockAny(overflow))
            {
                theta = 1;
                return false;
            }

            theta = blockMax(threadTheta);

            if (theta > 1)
            {
                return false;
            }
        }

        if (active)
        {
            ode->derivatives(tNew, resources->currentParameter(0), yTemp, dy);
            LUBacksubstitute(block.a, block.pivotIndices, dy, systemSize);
        }
    }

    if (active)
    {
        sumVec(y, yTemp, dy, systemSize);
    }

    return true;
}

__device__
void kodes::Seulex::extrapolate (const label k, const label systemSize, scalar* __restrict__ table, scalar* __restrict__ y)
{
    for (label j=k-1; j>0; j--)
    {
        for (label i=0; i<systemSize; i++)
        {
            table[INDEXMAT(i, j-1, systemSize)] =
                table[INDEXMAT(i, j, systemSize)] + kSeulexExtrapolationCoeff[k + j*kSeulexTableSize]*(table[INDEXMAT(i, j, systemSize)] - table[INDEXMAT(i, j-1, systemSize)]);
        }
    }

    for (label i=0; i<systemSize; i++)
    {
        y[INDEXVEC(i)] = table[INDEXMAT(i, 0, systemSize)] + kSeulexExtrapolationCoeff[k]*(table[INDEXMAT(i, 0, systemSize)] - y[INDEXVEC(i)]);
    }
}

__device__
scalar kodes::Seulex::step
(
    kodes::ODESystem* ode,
    kodes::DeviceResources* deviceResources,
    kodes::BlockStep& blockStep,
    const bool active,
    kodes::IntegratorControls controls
) const
{
    // safe: the table entry that made this method made these resources
    SeulexDeviceResources* resources =
        static_cast<SeulexDeviceResources*>(deviceResources);

    const label systemSize = resources->systemSize();

    // The same array pyJac's generated code calls shared_temp; its 4*blockDim.x
    // doubles come first and the block's own scratch, if the plan put it here
    // at all, behind them.
    extern __shared__ scalar kodesSharedScratch[];

    char* blockScratchBase = controls.blockScratchInShared
      ? reinterpret_cast<char*>(kodesSharedScratch + blockScratchOffset(blockDim.x))
      : resources->blockScratchSlot();

    const SeulexBlockScratch block =
        SeulexDeviceResources::blockScratch(blockScratchBase, systemSize);

    const scalar absTol = controls.absTol;
    const scalar relTol = controls.relTol;

    const scalar jacRedo = min(1e-4, relTol);

    // Both outlive this call and neither outlives the chunk: kTarg is the order
    // the block's previous step settled on, theta the contraction it measured.
    // They are read into registers here and written back at the end, since
    // every thread holds the same value of each and only their crossing of a
    // step boundary needs memory at all.
    label  kTarg = block.kTarg[0];
    scalar theta = block.theta[0];

    scalar* __restrict__ table = resources->table();
    scalar* __restrict__ dfdy  = resources->dfdy();

    scalar* __restrict__ y0    = resources->y0();
    scalar* __restrict__ ySequence = resources->ySequence();
    scalar* __restrict__ scale = resources->scale();

    scalar* __restrict__ y = resources->currentVector();

    // The block's stiffest system - see kJacobianAtStiffestLane. The driver
    // does not call a method unless some lane is active, so this is never -1.
    const label jacLane = kJacobianAtStiffestLane
      ? blockMaxActiveLane(active)
      : 0;

    const label jacTid = blockIdx.x*blockDim.x + jacLane;

    scalar dt = blockStep.deltaTTry;

    if (threadIdx.x == 0)
    {
        block.temp[0] = GREAT;
        block.dtOpt[0] = fabs(0.1*dt);
    }

    if (active)
    {
        copyVec(y0, y, systemSize);
    }

    if (blockStep.first || blockStep.prevReject)
    {
        theta = 2*jacRedo;
    }

    if (blockStep.first)
    {
        const scalar logTol = -log10(relTol + absTol)*0.6 + 0.5;
        kTarg = max(1, min(kSeulexMaxOrder - 1, label(logTol)));
    }

    if (active)
    {
        for (label i=0; i < systemSize; ++i)
        {
            scale[INDEXVEC(i)] = absTol + relTol*fabs(y[INDEXVEC(i)]);
        }
    }

    bool jacUpdated = false;

    if (theta > jacRedo)
    {
        if (threadIdx.x == jacLane)
        {
            ode->jacobian(blockStep.t, resources->currentParameter(0), y, dfdy);
        }

        jacUpdated = true;
    }

    // Covers both the Jacobian above and the two writes to block.temp and
    // block.dtOpt, all of which the whole block reads from here on.
    __syncthreads();

    label k;
    scalar dtNew = fabs(dt);
    bool firstk = true;

    while (firstk || blockStep.reject)
    {
        dt = blockStep.forward ? dtNew : -dtNew;
        firstk = false;
        blockStep.reject = false;

        if (fabs(dt) <= fabs(blockStep.t) * sqr(SMALL) && threadIdx.x == 0)
        {
            printf("step size underflow : %0.16f \n", dt);
        }

        scalar errOld = 0;

        for (k=0; k<=kTarg+1; k++)
        {
            bool success = seul
            (
                resources, block, ode, jacTid, active, blockStep.t, dt, k, theta
            );

            if (!success)
            {
                blockStep.reject = true;
                dtNew = fabs(dt)*kSeulexStepFactor5;
                break;
            }

            if (active)
            {
                if (k == 0)
                {
                    copyVec(y, ySequence, systemSize);
                }
                else
                {
                    for (label i=0; i<systemSize; ++i)
                    {
                        table[INDEXMAT(i, k-1, systemSize)] = ySequence[INDEXVEC(i)];
                    }
                }
            }

            if (k != 0)
            {
                scalar threadErr = 0;

                if (active)
                {
                    extrapolate(k, systemSize, table, y);

                    for (label i=0; i<systemSize; ++i)
                    {
                        scale[INDEXVEC(i)] = absTol + relTol*fabs(y0[INDEXVEC(i)]);
                        threadErr += sqr((y[INDEXVEC(i)] - table[INDEXMAT(i, 0, systemSize)])/scale[INDEXVEC(i)]);
                    }

                    threadErr = sqrt(threadErr/systemSize);
                }

                // The block accepts or rejects together, so the error that
                // decides is the worst of its systems. Every branch below is
                // therefore taken by all of its threads.
                const scalar err = blockMax(threadErr);

                if (err > 1/SMALL || (k > 1 && err >= errOld))
                {
                    blockStep.reject = true;
                    dtNew = fabs(dt)*kSeulexStepFactor5;
                    break;
                }
                errOld = min(4*err, 1.0);
                scalar expo = 1.0/(k + 1);
                scalar facmin = pow(kSeulexStepFactor3, expo);
                scalar fac;
                if (err == 0)
                {
                    fac = 1/facmin;
                }
                else
                {
                    fac = kSeulexStepFactor2/pow(err/kSeulexStepFactor1, expo);
                    fac = max(facmin/kSeulexStepFactor4, min(1/facmin, fac));
                }

                if (threadIdx.x == 0)
                {
                    block.dtOpt[k] = fabs(dt*fac);
                    block.temp[k] = kSeulexWorkEstimate[k]/block.dtOpt[k];
                }

                __syncthreads();

                if ((blockStep.first || blockStep.last) && err <= 1)
                {
                    break;
                }

                if
                (
                    k == kTarg - 1
                && !blockStep.prevReject
                && !blockStep.first && !blockStep.last
                )
                {
                    if (err <= 1)
                    {
                        break;
                    }
                    else if (err > kSeulexStepSequence[kTarg]*kSeulexStepSequence[kTarg + 1]*4)
                    {
                        blockStep.reject = true;
                        kTarg = k;
                        if (kTarg>1 && block.temp[k-1] < kSeulexKFactor1*block.temp[k])
                        {
                            kTarg--;
                        }
                        dtNew = block.dtOpt[kTarg];
                        break;
                    }
                }

                if (k == kTarg)
                {
                    if (err <= 1)
                    {
                        break;
                    }
                    else if (err > kSeulexStepSequence[k + 1]*2)
                    {
                        blockStep.reject = true;
                        if (kTarg>1 && block.temp[k-1] < kSeulexKFactor1*block.temp[k])
                        {
                            kTarg--;
                        }
                        dtNew = block.dtOpt[kTarg];
                        break;
                    }
                }

                if (k == kTarg+1)
                {
                    if (err > 1)
                    {
                        blockStep.reject = true;
                        if
                        (
                            kTarg > 1
                        && block.temp[kTarg-1] < kSeulexKFactor1*block.temp[kTarg]
                        )
                        {
                            kTarg--;
                        }
                        dtNew = block.dtOpt[kTarg];
                    }
                    break;
                }
            }
        }

        if (blockStep.reject)
        {
            blockStep.prevReject = true;

            if (!jacUpdated)
            {
                theta = 2*jacRedo;

                if (threadIdx.x == jacLane)
                {
                    ode->jacobian(blockStep.t, resources->currentParameter(0), y, dfdy);
                }

                __syncthreads();

                jacUpdated = true;
            }
        }
    }

    blockStep.deltaTDid = dt;
    blockStep.t += dt;

    label kopt;
    if (k == 1)
    {
        kopt = 2;
    }
    else if (k <= kTarg)
    {
        kopt=k;
        if (block.temp[k-1] < kSeulexKFactor1*block.temp[k])
        {
            kopt = k - 1;
        }
        else if (block.temp[k] < kSeulexKFactor2*block.temp[k - 1])
        {
            kopt = min(k + 1, kSeulexMaxOrder - 1);
        }
    }
    else
    {
        kopt = k - 1;
        if (k > 2 && block.temp[k-2] < kSeulexKFactor1*block.temp[k - 1])
        {
            kopt = k - 2;
        }
        if (block.temp[k] < kSeulexKFactor2*block.temp[kopt])
        {
            kopt = min(k, kSeulexMaxOrder - 1);
        }
    }

    if (blockStep.prevReject)
    {
        kTarg = min(kopt, k);
        dtNew = min(fabs(dt), block.dtOpt[kTarg]);
        blockStep.prevReject = false;
    }
    else
    {
        if (kopt <= k)
        {
            dtNew = block.dtOpt[kopt];
        }
        else
        {
            if (k < kTarg && block.temp[k] < kSeulexKFactor2*block.temp[k - 1])
            {
                dtNew = block.dtOpt[k]*kSeulexWorkEstimate[kopt + 1]/kSeulexWorkEstimate[k];
            }
            else
            {
                dtNew = block.dtOpt[k]*kSeulexWorkEstimate[kopt]/kSeulexWorkEstimate[k];
            }
        }
        kTarg = kopt;
    }

    blockStep.deltaTTry = blockStep.forward ? dtNew : -dtNew;

    // Carried to the next step of this chunk. The barrier is what makes the
    // read at the top of that step see them.
    if (threadIdx.x == 0)
    {
        block.kTarg[0] = kTarg;
        block.theta[0] = theta;
    }

    __syncthreads();

    // the step is already accepted and sized; nothing for adaptiveStep to judge
    return 0.0;
}
