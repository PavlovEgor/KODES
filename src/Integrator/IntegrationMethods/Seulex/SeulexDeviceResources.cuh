#ifndef KODES_SEULEX_DEVICE_RESOURCES
#define KODES_SEULEX_DEVICE_RESOURCES
#include "DeviceResources.cuh"
#include "ThreadMatrix.cuh"
#include "seulex_constants.cuh"
#include "ExtrapolationTable.cuh"

namespace kodes
{

// Everything added here is scratch space, sized with scratchSize: each thread
// integrates its own system, with its own Jacobian, its own factorization and
// its own extrapolation table.
class SeulexDeviceResources
    :
    public DeviceResources
{
protected:
    ExtrapolationTable table_;
    ThreadMatrix<scalar> dfdy_;
    ThreadMatrix<scalar> a_;

    ThreadVector<label> pivotIndices_;

    // Row scales of the implicit pivoting in LUDecompose. They used to be a
    // local scalar[128] there, which a mechanism of more than 128 species ran
    // past in silence - into the thread's stack.
    ThreadVector<scalar> vv_;

    ThreadVector<scalar> dtOpt_;
    ThreadVector<scalar> temp_;
    ThreadVector<scalar> y0_;
    ThreadVector<scalar> ySequence_;
    ThreadVector<scalar> scale_;
    ThreadVector<scalar> dy_;
    ThreadVector<scalar> yTemp_;
    ThreadVector<scalar> dydt_;

    // Order and Jacobian control carried from one step of a system to the next:
    // the target extrapolation order the previous step settled on, and the
    // contraction rate that decides whether the Jacobian may be reused. They
    // outlive a step but not a system, so they are scratch like everything
    // else here - a step re-reads what the previous one left in its slot.
    ThreadValue<label>  kTarg_;
    ThreadValue<scalar> theta_;

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

    // dtOpt_/temp_ are indexed by the extrapolation order k, not by the
    // component, so they need at least kSeulexTableSize entries per thread
    __host__ __device__ static label orderStorage(const label systemSize)
    {
        return systemSize > kSeulexTableSize ? systemSize : kSeulexTableSize;
    }

    // Device memory needed per resident thread
    __host__ static size_t scratchBytesPerThread(const label systemSize, const label parameterSize)
    {
        const size_t n = size_t(systemSize);

        return DeviceResources::scratchBytesPerThread(systemSize, parameterSize)
             + (ExtrapolationTable::kColumns * n    // table_
             + 2 * n * n                            // dfdy_, a_
             + 7 * n                                // vv_, y0_, ySequence_, scale_, dy_, yTemp_, dydt_
             + 2 * size_t(orderStorage(systemSize)) // dtOpt_, temp_
             + 1                                    // theta_
               ) * sizeof(scalar)
             + (n + 1) * sizeof(label);             // pivotIndices_, kTarg_
    }

    __device__ ExtrapolationTable
    table() const { return table_; }

    __device__ ThreadMatrix<scalar>
    dfdy() const { return dfdy_; }

    __device__ ThreadMatrix<scalar>
    a() const { return a_; }

    __device__ ThreadVector<label>
    pivotIndices() const { return pivotIndices_; }

    __device__ ThreadVector<scalar>
    vv() const { return vv_; }

    __device__ ThreadVector<scalar>
    dtOpt() const { return dtOpt_; }

    __device__ ThreadVector<scalar>
    temp() const { return temp_; }

    __device__ ThreadVector<scalar>
    y0() const { return y0_; }

    __device__ ThreadVector<scalar>
    ySequence() const { return ySequence_; }

    __device__ ThreadVector<scalar>
    scale() const { return scale_; }

    __device__ ThreadVector<scalar>
    dy() const { return dy_; }

    __device__ ThreadVector<scalar>
    yTemp() const { return yTemp_; }

    __device__ ThreadVector<scalar>
    dydt() const { return dydt_; }

    __device__ ThreadValue<label>
    kTarg() const { return kTarg_; }

    __device__ ThreadValue<scalar>
    theta() const { return theta_; }
};

}

#endif
