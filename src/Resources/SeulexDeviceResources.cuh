#include "DeviceResources.cuh"


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

