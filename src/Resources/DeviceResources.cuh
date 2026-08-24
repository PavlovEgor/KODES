#ifndef DEVRES
#define DEVRES

#include "Resources.cuh"
#include "StepState.cuh"

namespace kodes
{

// Device side storage of one batch.
//
// `vectors`/`parameters` hold the whole batch (state space, stride batchSize),
// while `y_`/`param_` - and every temporary an integrator adds on top of them -
// hold one system per *resident thread* (scratch space, stride scratchSize).
// Since scratchSize only has to cover the threads that can run at the same
// time, the d^2 sized temporaries stay small even for a batch of millions of
// systems.
class DeviceResources
    :
    public Resources,
    public StepState
{
public:

    scalar*        vectors;      // [systemSize][batchSize]
    scalar*        parameters;   // [parameterSize][batchSize]

protected:

    label          scratchSize_; // number of per thread slots

    scalar*        y_;           // [systemSize][scratchSize]
    scalar*        param_;       // [parameterSize][scratchSize]

public:

    __device__ __host__
    DeviceResources
    (
        const label batchSize,
        const label scratchSize,
        const label systemSize,
        const label parameterSize
    )
        : Resources(batchSize, systemSize, parameterSize),
          StepState(batchSize),
          scratchSize_(scratchSize)
    {}

    __device__ __host__
    ~DeviceResources() = default;

    __device__ static void* operator new(size_t size, void* ptr) {
        return ptr;
    }

    __host__ static DeviceResources*
    create(const label batchSize, const label scratchSize, const label systemSize, const label parameterSize);

    __host__ static void
    destroy(DeviceResources* devRes);

    __host__ __device__ void
    printVectori(const label i) const;

    __device__ __host__ label batchSize() const { return this->ensembleSize_; }

    __device__ __host__ label scratchSize() const { return scratchSize_; }

    // Working copy of the state of the system currently handled by this thread
    __device__ scalar* __restrict__ y() { return y_; }

    // Working copy of the parameters of that system
    __device__ scalar* __restrict__ param() { return param_; }

    // Pull system `system` of the batch into this thread's scratch slot.
    // Consecutive threads read consecutive addresses, so the transfer stays
    // coalesced.
    __device__ void loadSystem(const label system)
    {
        for (label i = 0; i < systemSize_; ++i)
        {
            y_[INDEXVEC(i)] = vectors[INDEXSTATE(system, i, batchSize_)];
        }

        for (label i = 0; i < parameterSize_; ++i)
        {
            param_[INDEXVEC(i)] = parameters[INDEXSTATE(system, i, batchSize_)];
        }
    }

    // Push this thread's scratch slot back into system `system` of the batch
    __device__ void storeSystem(const label system)
    {
        for (label i = 0; i < systemSize_; ++i)
        {
            vectors[INDEXSTATE(system, i, batchSize_)] = y_[INDEXVEC(i)];
        }
    }

    // Device memory needed per system of the batch
    __host__ static size_t stateBytesPerSystem(const label systemSize, const label parameterSize)
    {
        return size_t(systemSize + parameterSize) * sizeof(scalar)
             + StepState::bytesPerSystem();
    }

    // Device memory needed per resident thread
    __host__ static size_t scratchBytesPerThread(const label systemSize, const label parameterSize)
    {
        return size_t(systemSize + parameterSize) * sizeof(scalar);
    }
};

}

#endif
