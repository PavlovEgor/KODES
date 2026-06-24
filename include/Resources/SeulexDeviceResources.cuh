#include "DeviceResources.cuh"


namespace kodes 
{

class SeulexDeviceResources 
    :
    public DeviceResources
{
private:

public:

    __device__ __host__
    SeulexDeviceResources(const label numOfSystems, const label sizeOfSystem);

    __device__ __host__
    ~SeulexDeviceResources() override;

    __host__ void cpyHostToDevice(scalar** in_data_host) override;
    __host__ void cpyDeviceToHost(scalar** out_data_host) const override;
};

}

