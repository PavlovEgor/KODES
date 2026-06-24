#include "basic_types.cuh"

namespace kodes 
{

class DeviceResources 
{
private:
    label numOfSystems_;
    label sizeOfSystem_;
    label numOfParameters;

    scalar* resources_scalar;
    label*  resources_label;

    scalar* data_device;
    scalar* params_device;

public:
    virtual DeviceResources(const label numOfSystems, const label sizeOfSystem) =0;
    virtual ~DeviceResources() =0;

    virtual void cpyHostToDevice(const scalar** in_data_host) =0;
    virtual void cpyDeviceToHost(scalar** out_data_host) const =0;

};

}