#ifndef STEP_STATE
#define STEP_STATE

#include "basic_types.cuh"

namespace kodes
{
    class StepState
    {
    public: 

        label batchSize_;
        scalar deltaTMin;

        bool* forward;
        scalar* deltaTTry;
        scalar* deltaTDid;
        bool* first;
        bool* last;
        bool* reject;
        bool* prevReject;

        __device__
        StepState(label batchSize);

        __device__
        ~StepState();

        __device__
        void setDeltaT(const scalar deltaT);

        __device__ inline
        void setDeltaTMinToGreat()
        {
            if (INDEXVEC(0) == 0)
            {
                deltaTMin = GREAT;
            }
        }

        __device__
        scalar findMinDeltaT();
    };
}


#endif
