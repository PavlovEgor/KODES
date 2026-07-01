#include "DeviceResources.cuh"


namespace kodes 
{

class SeulexDeviceResources 
    :
    public DeviceResources 
{
protected:
    scalar* table_;
    scalar* dfdx_;
    scalar* dfdy_;
    scalar* a_;

    label* pivotIndices_;

    scalar* dxOpt_;
    scalar* temp_;
    scalar* y0_;
    scalar* ySequence_ ;
    scalar* scale_;
    scalar* dy_;
    scalar* yTemp_;
    scalar* dydx_;
    scalar* y_;

public:

    __device__ __host__
    SeulexDeviceResources(const label numOfSystems, const label sizeOfSystem, const label numOfParameters)
        : DeviceResources(numOfSystems, sizeOfSystem, numOfParameters) {}

    __device__ __host__
    ~SeulexDeviceResources() = default;

    __host__ static SeulexDeviceResources* 
    create(const label numOfSystems, const label sizeOfSystem, const label numOfParameters, SeulexDeviceResources* hostStub);

    __host__ static void
    destroy(SeulexDeviceResources* devRes, SeulexDeviceResources* hostStub);

    __device__ scalar* 
    table() { return table_; }

    __device__ scalar* 
    table(const size_t workIndex) { 
        return table_ + workIndex * 12 * (this -> sizeOfSystem_); 
    }

    __device__ scalar* 
    dfdx() { return dfdx_; }

    __device__ scalar* 
    dfdx(const size_t workIndex) { 
        return dfdx_ + workIndex * sizeOfSystem_; 
    }

    __device__ scalar* 
    dfdy() { return dfdy_; }

    __device__ scalar* 
    dfdy(const size_t workIndex) { 
        return dfdy_ + workIndex * sizeOfSystem_ * sizeOfSystem_; 
    }

    __device__ scalar* 
    a() { return a_; }

    __device__ scalar* 
    a(const size_t workIndex) { 
        return a_ + workIndex * sizeOfSystem_ * sizeOfSystem_; 
    }

    __device__ label* 
    pivotIndices() { return pivotIndices_; }

    __device__ label* 
    pivotIndices(const size_t workIndex) { 
        return pivotIndices_ + workIndex * sizeOfSystem_; 
    }

    __device__ scalar* 
    dxOpt() { return dxOpt_; }

    __device__ scalar* 
    dxOpt(const size_t workIndex) { 
        return dxOpt_ + workIndex * sizeOfSystem_; 
    }

    __device__ scalar* 
    temp() { return temp_; }

    __device__ scalar* 
    temp(const size_t workIndex) { 
        return temp_ + workIndex * sizeOfSystem_; 
    }

    __device__ scalar* 
    y0() { return y0_; }

    __device__ scalar* 
    y0(const size_t workIndex) { 
        return y0_ + workIndex * sizeOfSystem_; 
    }

    __device__ scalar* 
    ySequence() { return ySequence_; }

    __device__ scalar* 
    ySequence(const size_t workIndex) { 
        return ySequence_ + workIndex * sizeOfSystem_; 
    }

    __device__ scalar* 
    scale() { return scale_; }

    __device__ scalar* 
    scale(const size_t workIndex) { 
        return scale_ + workIndex * sizeOfSystem_; 
    }

    __device__ scalar* 
    dy() { return dy_; }

    __device__ scalar* 
    dy(const size_t workIndex) { 
        return dy_ + workIndex * sizeOfSystem_; 
    }

    __device__ scalar* 
    yTemp() { return yTemp_; }

    __device__ scalar* 
    yTemp(const size_t workIndex) { 
        return yTemp_ + workIndex * sizeOfSystem_; 
    }

    __device__ scalar* 
    dydx() { return dydx_; }

    __device__ scalar* 
    dydx(const size_t workIndex) { 
        return dydx_ + workIndex * sizeOfSystem_; 
    }

    __device__ scalar* 
    y() { return y_; }

    __device__ scalar* 
    y(const size_t workIndex) { 
        return y_ + workIndex * sizeOfSystem_; 
    }
};

}

