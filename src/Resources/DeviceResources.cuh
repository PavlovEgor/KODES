#ifndef KODES_DEVICE_RESOURCES
#define KODES_DEVICE_RESOURCES
#include "Resources.cuh"
#include "StepState.cuh"
#include "BatchArray.cuh"
#include "ThreadVector.cuh"
#include "device_object.cuh"

namespace kodes
{

// Device side storage of one batch, split in two address spaces:
//
//  * vectors/parameters hold the whole batch, one slot per system, stride
//    batchSize: each a BatchArray
//  * currentVector/currentParameters hold the system a thread is integrating
//    right now, one slot per resident thread, stride scratchSize: each a
//    ThreadVector - as are the step state and every temporary an integrator
//    adds on top of them
//
// Every thread integrates its own system on its own: its own step size, its
// own Jacobian, its own factorization. Since scratchSize only covers the
// threads that can run at the same time, the systemSize^2 temporaries stay
// small for a batch of millions of systems.
//
// The base of IntegrationMethod: a method is the storage it integrates in, and
// adds its own scratch to this as members. Until 1.2 a DeviceResources subclass
// was a separate object, chosen by the same row of the method table as the
// method that cast its pointer down to it.
//
// Never a row of a table itself, so no KODES_DECLARE_DEVICE_OBJECT: the four
// statics would be four host functions and two construction kernels nobody
// calls. Only the placement operator new below is inherited, and only the
// leaves - Seulex and Euler - declare the rest.
class DeviceResources
    :
    public Resources,
    public StepState
{
public:

    BatchArray<scalar, kSystemSize>    vectors;
    BatchArray<scalar, kParameterSize> parameters;

protected:

    label          scratchSize_;
    label          blocks_;

    ThreadVector<scalar>                 currentVector_;
    ThreadVector<scalar, kParameterSize> currentParameters_;

public:

    __device__ __host__
    DeviceResources
    (
        const label batchSize,
        const label scratchSize,
        const label blocks,
        const label systemSize,
        const label parameterSize
    )
        : Resources(batchSize, systemSize, parameterSize),
          StepState(),
          scratchSize_(scratchSize),
          blocks_(blocks)
    {}

    __device__ __host__
    ~DeviceResources() = default;

    __device__ static void* operator new(size_t size, void* ptr) { return ptr; }

    // The device buffers, allocated into this object's own views. Every
    // subclass calls this one first and then adds its own - see
    // Factory/deviceObject.cuh for who calls them and when.
    __host__ void allocate();

    __host__ void deallocate();

    __device__ void
    printVectori(const label i) const;

    __device__ __host__ label batchSize() const { return this->ensembleSize_; }

    __device__ __host__ label scratchSize() const { return scratchSize_; }

    __device__ __host__ label blocks() const { return blocks_; }

    __device__ ThreadVector<scalar> currentVector() const { return currentVector_; }

    __device__ scalar currentParameter(const label i) const
    {
        return currentParameters_[i];
    }

    __device__ scalar vectorComponent(const label system, const label i) const
    {
        return vectors(system, i);
    }

    // Consecutive threads touch consecutive systems, so both transfers between
    // the batch and the scratch slots stay coalesced.
    __device__ void loadSystem(const label system)
    {
        for (label i = 0; i < kSystemSize; ++i)
        {
            currentVector_[i] = vectors(system, i);
        }

        for (label i = 0; i < kParameterSize; ++i)
        {
            currentParameters_[i] = parameters(system, i);
        }
    }

    __device__ void storeSystem(const label system)
    {
        for (label i = 0; i < kSystemSize; ++i)
        {
            vectors(system, i) = currentVector_[i];
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

    // Nothing is shared by a block: every thread integrates on its own. Kept
    // because every row of a type table answers it, and the plan adds it up.
    __host__ static size_t scratchBytesPerBlock(const label systemSize, const label parameterSize)
    {
        return 0;
    }
};

}

#endif
