#include "basic_types.cuh"

namespace kodes 
{

template<class HostResourcesType, class DeviceResourcesType>
class Operator 
{
protected:

    HostResourcesType*       hostRes_;
    DeviceResourcesType*     deviceRes_;

    label                    ensembleSize_;
    label                    systemSize_;
    label                    parameterSize_;

    label                    batchSize_;
    label                    lastBatchSize_; 
    label                    lastBatchIndex_;

public:

    Operator(HostResourcesType* hostRes, DeviceResourcesType* deviceRes);
        
    virtual ~Operator() = default;

    virtual void cpyHostToDevice(label batchIndex);
    virtual void cpyDeviceToHost(label batchIndex);

    virtual label getRealBatchSize(label batchIndex);
};

}

#include "Operator.cu"
