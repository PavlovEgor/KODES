#ifndef STEP_STATE
#define STEP_STATE

#include <cuda_runtime.h>

#include "basic_types.cuh"

namespace kodes
{
    // Per system step bookkeeping, in state space: one entry per system of
    // the batch, indexed by the system and not by the thread
    class StepState
    {
    public:

        label batchSize_;

        // Minimum of deltaTTry over the whole ensemble, shared by every thread
        // of the grid. Reduced by findMinDeltaT(), reset by setDeltaTMinToGreat()
        scalar deltaTMin;

        bool* forward;
        scalar* deltaTTry;
        scalar* deltaTDid;
        scalar* currentT;
        bool* first;
        bool* last;
        bool* reject;
        bool* prevReject;

        __device__ __host__
        StepState(label batchSize);

        __device__ __host__
        ~StepState() = default;

        // The per system arrays are allocated from the host, the host side stub
        // owns them and create() copies the pointers into the device object
        __host__
        void allocate(const label batchSize);

        __host__
        void deallocate();

        // Device memory occupied by one system
        __host__ static size_t bytesPerSystem()
        {
            return 3 * sizeof(scalar) + 5 * sizeof(bool);
        }

        __device__
        void setDeltaT(const scalar deltaT, const label system);

        __device__
        void resetStep(const label system);
    };
}


#endif
