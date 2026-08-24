
#ifndef INTEG_CONTROLS
#define INTEG_CONTROLS
namespace kodes
{
class IntegratorControls
{
public:
    scalar absTol;
    scalar relTol;
    scalar Treact;
    label  maxSteps;

    scalar safeScale = 0.9;
    scalar alphaIncrease = 0.2;  
    scalar alphaDecrease = 0.25;
    scalar minScale = 0.2; 
    scalar maxScale = 10;

    label realBatchSize = 0;
    label batchIndex;

    // Index, within the current batch, of the system the calling thread is
    // integrating. Set by the solve kernel on every grid-stride iteration, for
    // diagnostics only - the working data is addressed by thread slot.
    label system = 0;

    scalar deltaT = 0;

    __device__ __host__
    IntegratorControls
    (
        const scalar absTol = 1e-12,
        const scalar relTol = 1e-1,
        const label maxSteps = 10000,
        const scalar Treact = 0
    )
    : absTol(absTol), relTol(relTol), Treact(Treact), maxSteps(maxSteps), batchIndex(-1) {}
};
}
#endif
