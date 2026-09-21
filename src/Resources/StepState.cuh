#ifndef KODES_STEP_STATE
#define KODES_STEP_STATE
#include <cuda_runtime.h>

#include "basic_types.cuh"

namespace kodes
{
    // The step a block is taking, in registers.
    //
    // Every field is the same in every thread of the block by construction:
    // each is either seeded identically at the top of the step or derived from
    // a block reduction, so no thread ever holds a different one and there is
    // nothing to put in shared memory. That is also what makes the branches on
    // these fields safe to sit around a __syncthreads().
    struct BlockStep
    {
        // local time of the systems the block holds, and the step it is trying
        scalar t;
        scalar deltaTTry;
        scalar deltaTDid;

        bool forward;
        bool first;
        bool last;
        bool reject;
        bool prevReject;

        __device__ void reset(const scalar deltaTTry)
        {
            this->t = 0.0;
            this->deltaTTry = deltaTTry;
            this->deltaTDid = 0.0;
            this->forward = deltaTTry > 0.0;
            this->first = true;
            this->last = false;
            this->reject = false;
            this->prevReject = false;
        }
    };

    // The block's trial step, and nothing else.
    //
    // A block integrates its share of the batch in lockstep - one step size,
    // one extrapolation order, one accept or reject for all of its threads -
    // so the rest of the step bookkeeping (the current time, the direction,
    // first/last/reject/prevReject) is the same in every thread of the block
    // by construction and lives in a register copy of BlockStep, not in device
    // memory. What cannot live there is the one value that has to survive the
    // kernel: the step the block ended a batch with is the step it starts the
    // next one with, and the next batch is another launch.
    class StepState
    {
    public:

        // One entry per block launched, addressed with B_ID
        scalar* deltaTTry;

        // Defaulted on purpose: it runs *after* the stub has been byte-copied
        // to the device, so anything it wrote to deltaTTry - a null included -
        // would overwrite the address the copy just brought. See
        // Factory/device_object.cuh.
        __device__ __host__
        StepState() = default;

        __device__ __host__
        ~StepState() = default;

        // Allocated from the host, into the host side stub; create() copies
        // the pointer into the device object.
        __host__
        void allocate(const label blocks);

        __host__
        void deallocate();

        __host__ static size_t bytesPerBlock()
        {
            return sizeof(scalar);
        }

        __device__
        void setDeltaT(const scalar deltaT);
    };
}


#endif
