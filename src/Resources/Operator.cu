namespace kodes 
{

template<class HostResourcesType, class DeviceResourcesType>
Operator<HostResourcesType, DeviceResourcesType>::Operator(HostResourcesType* hostRes, DeviceResourcesType* deviceRes)
: 
hostRes_(hostRes), 
deviceRes_(deviceRes), 
numOfSystems_(hostRes_->numOfSystems()),
systemSize_(hostRes_->systemSize()),
parameterSize_(hostRes_->parameterSize()),
batchSize_(deviceRes_->numOfSystems())
{
    lastBatchIndex_ = ((numOfSystems_ + batchSize_ - 1) / batchSize_) - 1;

    if ((numOfSystems_ / batchSize_) == lastBatchIndex_ + 1)
    {
        lastBatchSize_ = batchSize_;
    } else 
    {
        lastBatchSize_ = numOfSystems_ - (numOfSystems_ / batchSize_) * batchSize_;
    }

}

template<class HostResourcesType, class DeviceResourcesType>
void Operator<HostResourcesType, DeviceResourcesType>::cpyHostToDevice(label batchIndex)
{
    size_t dataSize = ((batchIndex == lastBatchIndex_) ? lastBatchSize_ : batchSize_) * sizeof(scalar);

    for (label i=0; i < systemSize_; i++)
    {
        cudaMemcpy(deviceRes_->vectors + i * batchSize_, hostRes_->vectors[i] + batchIndex * batchSize_, dataSize, cudaMemcpyHostToDevice);
    }

    for (label i=0; i < parameterSize_; i++)
    {
        cudaMemcpy(deviceRes_->parameters + i * batchSize_, hostRes_->parameters[i] + batchIndex * batchSize_, dataSize, cudaMemcpyHostToDevice);
    }
}

template<class HostResourcesType, class DeviceResourcesType>
void Operator<HostResourcesType, DeviceResourcesType>::cpyDeviceToHost(label batchIndex)
{
    size_t dataSize = ((batchIndex == lastBatchIndex_) ? lastBatchSize_ : batchSize_) * sizeof(scalar);

    for (label i=0; i < systemSize_; i++)
    {
        cudaMemcpy(hostRes_->vectors[i] + batchIndex * batchSize_, deviceRes_->vectors + i * batchSize_, dataSize, cudaMemcpyDeviceToHost);
    }

    for (label i=0; i < parameterSize_; i++)
    {
        cudaMemcpy(hostRes_->parameters[i] + batchIndex * batchSize_, deviceRes_->parameters + i * batchSize_, dataSize, cudaMemcpyDeviceToHost);
    }
}

template<class HostResourcesType, class DeviceResourcesType>
label Operator<HostResourcesType, DeviceResourcesType>::getRealBatchSize(label batchIndex)
{
    return (batchIndex == lastBatchIndex_) ? lastBatchSize_ : batchSize_;
}
}