#ifndef KODES_DEVICE_RESOURCES
#define KODES_DEVICE_RESOURCES
#include "Resources.cuh"
#include "StepState.cuh"
#include "device_object.cuh"

namespace kodes
{

// Device side storage of one batch, split in three address spaces:
//
//  * vectors/parameters hold the whole batch, one slot per system, stride
//    batchSize, addressed with INDEXSTATE
//  * currentVector/currentParameters hold the system a thread is integrating
//    right now, one slot per resident thread, stride scratchSize, addressed
//    with INDEXVEC - as is every per system temporary an integrator adds
//  * the block scratch holds what a whole block shares while it integrates its
//    threads' systems in lockstep: the matrix it factorises once, its pivots,
//    and the step control that is one value for all of them. One slot per
//    block, addressed with INDEXBLOCKMAT and INDEXBLOCKVEC
//
// The last of the three is why the systemSize^2 work matrix costs one copy per
// block rather than one per resident thread, and why for a small mechanism it
// fits in shared memory at all.
//
// The abstract base of a device object family, like Balancer and ODESystem:
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

    // Traversal order of the batch, owned by a Balancer. Null means the
    // systems are taken in the order they were copied in.
    const label*   order_;

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
          blocks_(blocks),
          order_(nullptr)
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
        return size_t(systemSize + parameterSize) * sizeof(scalar);
    }

    // What one block needs on top of its threads' slots. Only the trial step
    // here; a method that shares a matrix adds it in its own subclass.
    //
    // This figure is also what the plan compares against the shared memory a
    // block may have (see blockScratchInShared in LaunchConfig.cuh), although
    // the trial step counted here can never move there - it has to survive the
    // launch. The slack is one scalar per block: it makes the plan marginally
    // more reluctant to use shared memory, never less, which is the safe
    // direction.
    __host__ static size_t scratchBytesPerBlock(const label systemSize, const label parameterSize)
    {
        return StepState::bytesPerBlock();
    }
};

}

#endif
