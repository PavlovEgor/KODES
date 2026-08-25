#ifndef KODES_DEVICE_RESOURCES
#define KODES_DEVICE_RESOURCES
#include "Resources.cuh"
#include "StepState.cuh"
#include "deviceObject.cuh"

namespace kodes
{

// Device side storage of one batch, split in two address spaces:
//
//  * vectors/parameters hold the whole batch, one slot per system, stride
//    batchSize, addressed with INDEXSTATE
//  * currentVector/currentParameters hold the system a thread is integrating
//    right now, one slot per resident thread, stride scratchSize, addressed
//    with INDEXVEC - as are the step state and every temporary an integrator
//    adds on top of them
//
// Since scratchSize only covers the threads that can run at the same time, the
// systemSize^2 temporaries stay small for a batch of millions of systems.
class DeviceResources
    :
    public Resources,
    public StepState
{
public:

    scalar*        vectors;
    scalar*        parameters;

protected:

    label          scratchSize_;

    scalar*        currentVector_;
    scalar*        currentParameters_;

    // Traversal order of the batch, owned by a Balancer. Null means the
    // systems are taken in the order they were copied in.
    const label*   order_;

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
          StepState(),
          scratchSize_(scratchSize),
          order_(nullptr)
    {}

    __device__ __host__
    ~DeviceResources() = default;

    KODES_DECLARE_DEVICE_OBJECT(DeviceResources)

    // The device buffers, allocated into this object's own pointers. Every
    // subclass calls this one first and then adds its own - see
    // Factory/deviceObject.cuh for who calls them and when.
    __host__ void allocate();

    __host__ void deallocate();

    __host__ __device__ void
    printVectori(const label i) const;

    __device__ __host__ label batchSize() const { return this->ensembleSize_; }

    __device__ __host__ label scratchSize() const { return scratchSize_; }

    __device__ scalar* __restrict__ currentVector() { return currentVector_; }

    __device__ scalar currentParameter(const label i) const
    {
        return currentParameters_[INDEXVEC(i)];
    }

    __device__ void useOrder(const label* order) { order_ = order; }

    // Index of the system sitting at position i of the balanced traversal
    __device__ label systemAt(const label i) const { return order_ ? order_[i] : i; }

    __device__ scalar vectorComponent(const label system, const label i) const
    {
        return vectors[INDEXSTATE(system, i, batchSize())];
    }

    // Consecutive threads touch consecutive systems, so both transfers between
    // the batch and the scratch slots stay coalesced.
    __device__ void loadSystem(const label system)
    {
        for (label i = 0; i < systemSize_; ++i)
        {
            currentVector_[INDEXVEC(i)] = vectors[INDEXSTATE(system, i, batchSize())];
        }

        for (label i = 0; i < parameterSize_; ++i)
        {
            currentParameters_[INDEXVEC(i)] = parameters[INDEXSTATE(system, i, batchSize())];
        }
    }

    __device__ void storeSystem(const label system)
    {
        for (label i = 0; i < systemSize_; ++i)
        {
            vectors[INDEXSTATE(system, i, batchSize())] = currentVector_[INDEXVEC(i)];
        }
    }

    __host__ static size_t stateBytesPerSystem(const label systemSize, const label parameterSize)
    {
        return size_t(systemSize + parameterSize) * sizeof(scalar);
    }

    __host__ static size_t scratchBytesPerThread(const label systemSize, const label parameterSize)
    {
        return size_t(systemSize + parameterSize) * sizeof(scalar)
             + StepState::bytesPerThread();
    }
};

}

#endif
