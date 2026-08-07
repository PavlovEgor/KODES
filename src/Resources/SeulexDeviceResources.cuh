#ifndef SEULEXDEVRES 
#define SEULEXDEVRES
#include "DeviceResources.cuh"

#include <array>

namespace kodes 
{

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
    scalar* y_;

public:

    static constexpr label  kMaxx_ = 12,
                            iMaxx_ = kMaxx_ + 1;

    static constexpr scalar stepFactor1_ = 0.6,
                            stepFactor2_ = 0.93,
                            stepFactor3_ = 0.1,
                            stepFactor4_ = 4,
                            stepFactor5_ = 0.5,
                            kFactor1_ = 0.7,
                            kFactor2_ = 0.9;

    label nSeq_[iMaxx_];
    scalar gpu_[iMaxx_];
    scalar coeff_[iMaxx_ * iMaxx_];

    __device__ __host__
    SeulexDeviceResources(const label ensembleSize, const label systemSize, const label parameterSize)
        : DeviceResources(ensembleSize, systemSize, parameterSize) {}

    __device__ __host__
    ~SeulexDeviceResources() = default;

    __host__ static SeulexDeviceResources* 
    create(const label ensembleSize, const label systemSize, const label parameterSize, SeulexDeviceResources* hostStub);

    __host__ static void
    destroy(SeulexDeviceResources* devRes, SeulexDeviceResources* hostStub);

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

    __device__ scalar* 
    y() { return y_; }

};

}

#define kMaxx_ (12)
#define iMaxx_ (kMaxx_ + 1)

__constant__ scalar stepFactor1_ = 0.6;
__constant__ scalar stepFactor2_ = 0.4;
__constant__ scalar stepFactor3_ = 0.3;
__constant__ scalar stepFactor4_ = 0.2;
__constant__ scalar stepFactor5_ = 0.1;
__constant__ scalar kFactor1_ = 0.5;
__constant__ scalar kFactor2_ = 0.5;

__constant__ label nSeq_[iMaxx_];
__constant__ scalar gpu_[iMaxx_];
__constant__ scalar coeff_[iMaxx_ * iMaxx_];


#endif 
