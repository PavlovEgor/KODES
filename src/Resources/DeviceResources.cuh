#ifndef KODES_DEVICE_RESOURCES
#define KODES_DEVICE_RESOURCES
#include "Resources.cuh"
#include "StepState.cuh"
#include "device_object.cuh"

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
// Every thread integrates its own system on its own: its own step size, its
// own Jacobian, its own factorization. Since scratchSize only covers the
// threads that can run at the same time, the systemSize^2 temporaries stay
// small for a batch of millions of systems.
//
// The abstract base of a device object family, like ODESystem:
// the subclasses keep the same three-point contract and are built by the same
// four statics. What it does *not* have, and must not, is a table of its own.
//
// A method and the resources holding its scratch can only be chosen together -
// Seulex::step() casts this pointer down to SeulexDeviceResources*, Euler::step()
// to EulerDeviceResources*, IntegrationMethod::adaptiveStep() to
// AdaptiveDeviceResources* - so the row that selects a subclass is a field of
// the method's row, in IntegrationMethods/method_table.cu. Were it a table with
// a name of its own, a settings file naming "euler" and "seulexResources"
// would pass every check and fail inside the kernel.
//
// Which is also why this class and AdaptiveDeviceResources carry no
// KODES_DECLARE_DEVICE_OBJECT: neither is ever a row, so the four statics would
// be four host functions and two construction kernels nobody calls. Only the
// placement operator new below is inherited, and only the leaves - Seulex's
// and Euler's - declare the rest.
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
    label          blocks_;

    scalar*        currentVector_;
    scalar*        currentParameters_;

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

    // The device buffers, allocated into this object's own pointers. Every
    // subclass calls this one first and then adds its own - see
    // Factory/deviceObject.cuh for who calls them and when.
    __host__ void allocate();

    __host__ void deallocate();

    __host__ __device__ void
    printVectori(const label i) const;

    __device__ __host__ label batchSize() const { return this->ensembleSize_; }

    __device__ __host__ label scratchSize() const { return scratchSize_; }

    __device__ __host__ label blocks() const { return blocks_; }

    __device__ scalar* __restrict__ currentVector() { return currentVector_; }

    __device__ scalar currentParameter(const label i) const
    {
        return currentParameters_[INDEXVEC(i)];
    }

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

    // Nothing is shared by a block: every thread integrates on its own. Kept
    // because every row of a type table answers it, and the plan adds it up.
    __host__ static size_t scratchBytesPerBlock(const label systemSize, const label parameterSize)
    {
        return 0;
    }
};

}

#endif
