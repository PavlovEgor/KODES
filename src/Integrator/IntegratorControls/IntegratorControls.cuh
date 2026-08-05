
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

    label realBatchSize;
    label batchIndex;

    scalar deltaT;

    __device__ __host__
    IntegratorControls
    (
        const scalar absTol = 1e-12,
        const scalar relTol = 1e-1,
        const label maxSteps = 10000,
        const scalar Treact = 0
    )
    : absTol(absTol), relTol(relTol), maxSteps(maxSteps), Treact(Treact), batchIndex(-1) {}
};
}
#endif
