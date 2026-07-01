#include "Operator.cuh"

namespace kodes 
{

template<class HostResourcesType, class DeviceResourcesType>
Operator<HostResourcesType, DeviceResourcesType>::Operator(HostResourcesType* hostRes, DeviceResourcesType* deviceRes)
: hostRes_(hostRes), deviceRes_(deviceRes) 
{
    // if (hostRes->numOfSystems_ != deviceRes -> numOfSystems_)
    // {
    //     printf("Wrong numOfSystems. hostRes->numOfSystems = %d, deviceRes -> numOfSystems = %d", hostRes->numOfSystems_, deviceRes -> numOfSystems_);
    // }
    // if (hostRes->sizeOfSystem_ != deviceRes -> sizeOfSystem_)
    // {
    //     printf("Wrong sizeOfSystem. hostRes->sizeOfSystem = %d, deviceRes -> sizeOfSystem = %d", hostRes->sizeOfSystem_, deviceRes -> sizeOfSystem_);
    // }
    // if (hostRes->numOfParameters_ != deviceRes -> numOfParameters_)
    // {
    //     printf("Wrong numOfParameters. hostRes->numOfParameters = %d, deviceRes -> numOfParameters = %d", hostRes->numOfParameters_, deviceRes -> numOfParameters_);
    // }
}

template<class HostResourcesType, class DeviceResourcesType>
void Operator<HostResourcesType, DeviceResourcesType>::cpyHostToDevice()
{
    for (label i=0; i < hostRes_->sizeOfSystem_; i++)
    {
        cudaMemcpy(deviceRes_->vectors + i * hostRes_->numOfSystems_, hostRes_->vectors[i], hostRes_->numOfSystems_ * sizeof(scalar), cudaMemcpyHostToDevice);
    }

    for (label i=0; i < hostRes_->numOfParameters_; i++)
    {
        cudaMemcpy(deviceRes_->parameters + i * hostRes_->numOfSystems_, hostRes_->parameters[i], hostRes_->numOfSystems_ * sizeof(scalar), cudaMemcpyHostToDevice);
    }
}

template<class HostResourcesType, class DeviceResourcesType>
void Operator<HostResourcesType, DeviceResourcesType>::cpyDeviceToHost()
{
    for (label i=0; i < hostRes_->sizeOfSystem_; i++)
    {
        cudaMemcpy(hostRes_->vectors[i], deviceRes_->vectors + i * hostRes_->numOfSystems_, hostRes_->numOfSystems_ * sizeof(scalar), cudaMemcpyDeviceToHost);
    }

    for (label i=0; i < hostRes_->numOfParameters_; i++)
    {
        cudaMemcpy(hostRes_->parameters[i], deviceRes_->parameters + i * hostRes_->numOfSystems_, hostRes_->numOfSystems_ * sizeof(scalar), cudaMemcpyDeviceToHost);
    }
}
}