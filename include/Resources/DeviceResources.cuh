#include "basic_types.cuh"

namespace kodes 
{

class DeviceResources 
{
protected:
    label numOfSystems_;
    label sizeOfSystem_;
    label numOfParameters_;

    scalar* resources_scalar;
    label*  resources_label;

    scalar* data_device;
    scalar* params_device;

public:

    __device__ __host__
    DeviceResources(const label numOfSystems, const label sizeOfSystem) 
        : numOfSystems_(numOfSystems), sizeOfSystem_(sizeOfSystem) {}
        
    __device__ __host__
    virtual ~DeviceResources() = default;

    virtual void cpyHostToDevice(scalar** in_data_host) =0;
    virtual void cpyDeviceToHost(scalar** out_data_host) const =0;

    __device__ __host__
    scalar* getDataPrt() {return data_device; }
    scalar* getParamPrt(){return params_device;}

    
    template<typename T>
    T* getResources();
};

}