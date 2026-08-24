#ifndef SEULEXDEVRES
#define SEULEXDEVRES
#include "DeviceResources.cuh"
#include "SeulexConstants.cuh"

#include <array>

namespace kodes
{

// All the arrays added here are scratch space: they are sized with
// scratchSize (the number of threads that can be resident at the same time),
// *not* with the batch size, and are indexed with INDEXVEC/INDEXMAT.
class SeulexDeviceResources
    :
    public DeviceResources
{
protected:
    scalar* table_;
    scalar* dfdt_;
    scalar* dfdy_;
    scalar* a_;

    label* pivotIndices_;

    scalar* dtOpt_;
    scalar* temp_;
    scalar* y0_;
    scalar* ySequence_ ;
    scalar* scale_;
    scalar* dy_;
    scalar* yTemp_;
    scalar* dydt_;

public:

    __device__ __host__
    SeulexDeviceResources
    (
        const label batchSize,
        const label scratchSize,
        const label systemSize,
        const label parameterSize
    )
        : DeviceResources(batchSize, scratchSize, systemSize, parameterSize) {}

    __device__ __host__
    ~SeulexDeviceResources() = default;

    __host__ static SeulexDeviceResources*
    create
    (
        const label batchSize,
        const label scratchSize,
        const label systemSize,
        const label parameterSize,
        SeulexDeviceResources* hostStub
    );

    __host__ static void
    destroy(SeulexDeviceResources* devRes, SeulexDeviceResources* hostStub);

    // dtOpt_/temp_ are indexed by the extrapolation order k, not by the
    // component, so they need at least iMaxx_ entries per thread
    __host__ __device__ static label orderStorage(const label systemSize)
    {
        return systemSize > iMaxx_ ? systemSize : iMaxx_;
    }

    // Device memory needed per resident thread
    __host__ static size_t scratchBytesPerThread(const label systemSize, const label parameterSize)
    {
        const size_t n = size_t(systemSize);

        return DeviceResources::scratchBytesPerThread(systemSize, parameterSize)
             + (12 * n                              // table_
             + 2 * n * n                            // dfdy_, a_
             + 7 * n                                // dfdt_, y0_, ySequence_, scale_, dy_, yTemp_, dydt_
             + 2 * size_t(orderStorage(systemSize)) // dtOpt_, temp_
               ) * sizeof(scalar)
             + n * sizeof(label);                   // pivotIndices_
    }

    __device__ scalar*
    table() { return table_; }

    __device__ scalar*
    dfdt() { return dfdt_; }

    __device__ scalar*
    dfdy() { return dfdy_; }

    __device__ scalar*
    a() { return a_; }

    __device__ label*
    pivotIndices() { return pivotIndices_; }

    __device__ scalar*
    dtOpt() { return dtOpt_; }

    __device__ scalar*
    temp() { return temp_; }

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
