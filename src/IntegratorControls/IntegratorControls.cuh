
#ifndef INTEG_CONTROLS
#define INTEG_CONTROLS
namespace kodes
{
class IntegratorControls
{
public:
    scalar  absTol;
    scalar  relTol;
    label   maxSteps;

    __device__ __host__
    IntegratorControls
    (
        const scalar absTol = 1e-12,
        const scalar relTol = 1e-1,
        const label maxSteps = 10000
    )
    : absTol(absTol), relTol(relTol), maxSteps(maxSteps) {}
};
}
#endif
