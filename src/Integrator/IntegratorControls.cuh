
#ifndef KODES_INTEGRATOR_CONTROLS
#define KODES_INTEGRATOR_CONTROLS
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

    // First position of the balanced order the calling block is working on.
    // Set by the solve kernel on every grid-stride iteration, for diagnostics
    // only - the working data is addressed by thread and by block slot.
    label position = 0;

    scalar deltaT = 0;

    // Whether the scratch a block shares is worked in dynamic shared memory
    // or in the block's slot of the global buffer. Decided by the plan, copied
    // in from the LaunchConfig by Integrator::solve.
    bool blockScratchInShared = false;

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
