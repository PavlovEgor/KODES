#include "resources_handler.cuh"


namespace kodes 
{

class SeulexDeviceResources 
    :
    public DeviceResources
{
private:

public:
    virtual SeulexDeviceResources(const label numOfSystems, const label sizeOfSystem) =0;
    virtual ~SeulexDeviceResources() =0;

    void cpyHostToDevice(const scalar** in_data_host);
    void cpyDeviceToHost(scalar** out_data_host) const;
};

}

