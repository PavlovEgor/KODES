#ifndef STEP_STATE
#define STEP_STATE

#include "basic_types.cuh"

namespace kodes
{
    // CUDA provides atomicMin only for integer types, for scalar it is
    // emulated with a compare-and-swap loop
    __device__ inline
    void atomicMinScalar(scalar* address, const scalar value)
    {
        static_assert
        (
            sizeof(scalar) == sizeof(unsigned long long int),
            "atomicMinScalar expects a 64 bit scalar"
        );

        unsigned long long int* addressAsULL = (unsigned long long int*)address;
        unsigned long long int old = *addressAsULL;

        while (value < __longlong_as_double(old))
        {
            const unsigned long long int assumed = old;

            old = atomicCAS(addressAsULL, assumed, __double_as_longlong(value));

            if (old == assumed)
            {
                break;
            }
        }
    }

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
