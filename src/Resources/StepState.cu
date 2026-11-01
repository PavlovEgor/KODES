#include "StepState.cuh"

__host__
void kodes::StepState::allocate(const label scratchSize)
{
    forward.allocate(scratchSize);
    deltaTTry.allocate(scratchSize);
    deltaTDid.allocate(scratchSize);
    currentT.allocate(scratchSize);
    first.allocate(scratchSize);
    last.allocate(scratchSize);
}

__host__
void kodes::StepState::deallocate()
{
    forward.deallocate();
    deltaTTry.deallocate();
    deltaTDid.deallocate();
    currentT.deallocate();
    first.deallocate();
    last.deallocate();
}

__device__
void kodes::StepState::setDeltaT(const scalar deltaT)
{
    forward = deltaT > 0.0;
    deltaTTry = deltaT;
    deltaTDid = 0.0;
    currentT = 0.0;
    first = true;
    last = false;
}

__device__
void kodes::StepState::resetStep()
{
    setDeltaT(deltaTTry);
}
