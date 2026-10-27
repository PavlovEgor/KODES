#ifndef KODES_STEP_STATE
#define KODES_STEP_STATE
#include <cuda_runtime.h>

#include "basic_types.cuh"

namespace kodes
{
    // Step bookkeeping of the system a thread is integrating right now. Scratch
    // space: one entry per resident thread, addressed with INDEXVEC(0). Nothing
    // here outlives a system - resetStep() reseeds it from the trial step the
    // slot ended the previous system with.
    class StepState
    {
    public:

        bool* forward;
        scalar* deltaTTry;
        scalar* deltaTDid;
        scalar* currentT;
        bool* first;
        bool* last;
        bool* reject;
        bool* prevReject;

        // Empty, and written out rather than `= default`, which is not the
        // same thing here.
        //
        // This constructor runs *after* the stub has been byte-copied to the
        // device, so anything it wrote to the pointers above - a null
        // included - would overwrite the addresses the copy just brought (see
        // Factory/device_object.cuh). A `= default` constructor is not
        // user-provided, and DeviceResources names this base in its
        // mem-initializer list as `StepState()`, which for a class with a
        // non-user-provided default constructor is *value*-initialization: it
        // zeroes the object first. An empty body is default-initialization
        // instead, and leaves them alone.
        __device__ __host__
        StepState() {}

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
