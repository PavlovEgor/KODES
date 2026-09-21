#ifndef KODES_SEULEX_DEVICE_RESOURCES
#define KODES_SEULEX_DEVICE_RESOURCES
#include "DeviceResources.cuh"
#include "seulex_constants.cuh"

namespace kodes
{

// Everything a block shares while it takes one step, carved out of a single
// region so that the region can be either the block's slot of a global buffer
// or the dynamic shared memory behind pyJac's shared_temp - the method cannot
// tell which, and does not want to.
struct SeulexBlockScratch
{
    // The work matrix I/dt - J, factorised once for the whole block
    scalar* a;

    // Row scales of the factorization's implicit pivoting
    scalar* vv;

    // Indexed by the extrapolation order, not by the component: the step size
    // each order would like next and the work it costs to get there
    scalar* dtOpt;
    scalar* temp;

    // Contraction rate the block measured, which decides whether the Jacobian
    // may be reused on the next step
    scalar* theta;

    label*  pivotIndices;

    // Target extrapolation order the block's previous step settled on
    label*  kTarg;
};

// Scratch is in two parts here. Per thread: the state of the system that
// thread is integrating and the temporaries that go with it. Per block: the
// Jacobian's work matrix and the step control, because a block integrates its
// systems in lockstep - one step size, one order, one factorization.
//
// dfdy is the exception on both counts. It stays per thread although only one
// thread of the block fills it, because pyJac's generated eval_jacob writes
// through its own INDEX macro, whose GRID_DIM stride no base pointer can bend
// into a per block slot. The block reads it back out of that one thread's slot
// with INDEXJACMAT.
class SeulexDeviceResources
    :
    public DeviceResources
{
protected:
    scalar* table_;
    scalar* dfdy_;

    scalar* y0_;
    scalar* ySequence_ ;
    scalar* scale_;
    scalar* dy_;
    scalar* yTemp_;
    scalar* dydt_;

    // blocks_ slots of blockScratchBytes(systemSize_) each. Left unused when
    // the plan put the block scratch in shared memory instead, which it only
    // does when the region is small enough to fit there in the first place.
    char* blockScratch_;

public:

    __device__ __host__
    SeulexDeviceResources
    (
        const label batchSize,
        const label scratchSize,
        const label blocks,
        const label systemSize,
        const label parameterSize
    )
        : DeviceResources(batchSize, scratchSize, blocks, systemSize, parameterSize) {}

    __device__ __host__
    ~SeulexDeviceResources() = default;

    KODES_DECLARE_DEVICE_OBJECT(SeulexDeviceResources)

    __host__ void allocate();

    __host__ void deallocate();

    // Size of one block's region. Scalars first, labels after, so that the
    // region needs no padding inside it; the total is rounded up so that the
    // next block's slot starts aligned for a scalar again.
    __host__ __device__ static size_t blockScratchBytes(const label systemSize)
    {
        const size_t n = size_t(systemSize);

        const size_t bytes =
            (n*n + n + 2*size_t(kSeulexTableSize) + 1) * sizeof(scalar)
          + (n + 1) * sizeof(label);

        return (bytes + sizeof(scalar) - 1) / sizeof(scalar) * sizeof(scalar);
    }

    // Device memory needed per resident thread
    __host__ static size_t scratchBytesPerThread(const label systemSize, const label parameterSize)
    {
        const size_t n = size_t(systemSize);

        return DeviceResources::scratchBytesPerThread(systemSize, parameterSize)
             + (12 * n     // table_
             + n * n       // dfdy_
             + 6 * n       // y0_, ySequence_, scale_, dy_, yTemp_, dydt_
               ) * sizeof(scalar);
    }

    __host__ static size_t scratchBytesPerBlock(const label systemSize, const label parameterSize)
    {
        return DeviceResources::scratchBytesPerBlock(systemSize, parameterSize)
             + blockScratchBytes(systemSize);
    }

    // Carve one block's region up. `base` is the block's slot of the global
    // buffer or the shared memory the plan chose instead - see
    // IntegratorControls::blockScratchInShared.
    __device__ static SeulexBlockScratch
    blockScratch(char* base, const label systemSize)
    {
        const size_t n = size_t(systemSize);

        scalar* s = reinterpret_cast<scalar*>(base);

        SeulexBlockScratch block;

        block.a      = s;                       s += n*n;
        block.vv     = s;                       s += n;
        block.dtOpt  = s;                       s += kSeulexTableSize;
        block.temp   = s;                       s += kSeulexTableSize;
        block.theta  = s;                       s += 1;

        label* l = reinterpret_cast<label*>(s);

        block.pivotIndices = l;                 l += n;
        block.kTarg        = l;

        return block;
    }

    // The calling block's slot of the global buffer
    __device__ char* blockScratchSlot()
    {
        return blockScratch_ + size_t(B_ID) * blockScratchBytes(systemSize_);
    }

    __device__ scalar*
    table() { return table_; }

    __device__ scalar*
    dfdy() { return dfdy_; }

    __device__ scalar*
    y0() { return y0_; }

    __device__ scalar*
    ySequence() { return ySequence_; }

    __device__ scalar*
    scale() { return scale_; }

    __device__ scalar*
    dy() { return dy_; }

    __device__ scalar*
    yTemp() { return yTemp_; }

    __device__ scalar*
    dydt() { return dydt_; }

};

}

#endif
