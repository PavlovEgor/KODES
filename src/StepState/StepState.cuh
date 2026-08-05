#ifndef STEP_STATE
#define STEP_STATE

#include <cuda_runtime.h>

#include "basic_types.cuh"

namespace kodes
{
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

        __device__
        void setDeltaT(const scalar deltaT);

        __device__
        void resetStep();
    };
}


#endif
