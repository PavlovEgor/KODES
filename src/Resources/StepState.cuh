#ifndef KODES_STEP_STATE
#define KODES_STEP_STATE
#include <cuda_runtime.h>

#include "basicTypes.cuh"

namespace kodes
{
    // Step bookkeeping of the system a thread is integrating right now. Scratch
    // space: one entry per resident thread, addressed with INDEXVEC(0). Nothing
    // here outlives a system - resetStep() reseeds it from the trial step the
    // slot ended the previous system with.
    class StepState
    {
    public:

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
        StepState();

        __device__ __host__
        ~StepState() = default;

        // The per thread arrays are allocated from the host, the host side stub
        // owns them and create() copies the pointers into the device object
        __host__
        void allocate(const label scratchSize);

        __host__
        void deallocate();

        __host__ static size_t bytesPerThread()
        {
            return 3 * sizeof(scalar) + 5 * sizeof(bool);
        }

        __device__
        void setDeltaT(const scalar deltaT);

        __device__
        void resetStep();
    };
}


#endif
