namespace kodes 
{

template<class HostResourcesType, class DeviceResourcesType>
Operator<HostResourcesType, DeviceResourcesType>::Operator(HostResourcesType* hostRes, DeviceResourcesType* deviceRes)
: 
hostRes_(hostRes), 
deviceRes_(deviceRes), 
ensembleSize_(hostRes_->ensembleSize()),
systemSize_(hostRes_->systemSize()),
parameterSize_(hostRes_->parameterSize()),
batchSize_(deviceRes_->ensembleSize())
{
    if (!hostRes_ || !deviceRes_)
    {
        fprintf(stderr, "Operator ctor error at %s:%d: null resources pointer\n", __FILE__, __LINE__);
        std::exit(EXIT_FAILURE);
    }

    if (batchSize_ <= 0)
    {
        fprintf(stderr, "Operator ctor error at %s:%d: batchSize_ <= 0\n", __FILE__, __LINE__);
        std::exit(EXIT_FAILURE);
    }

    lastBatchIndex_ = ((ensembleSize_ + batchSize_ - 1) / batchSize_) - 1;

    if ((ensembleSize_ / batchSize_) == lastBatchIndex_ + 1)
    {
        lastBatchSize_ = batchSize_;
    } else 
    {
        lastBatchSize_ = ensembleSize_ - (ensembleSize_ / batchSize_) * batchSize_;
    }

}

template<class HostResourcesType, class DeviceResourcesType>
void Operator<HostResourcesType, DeviceResourcesType>::cpyHostToDevice(label batchIndex)
{
    if (batchIndex < 0 || batchIndex > lastBatchIndex_)
    {
        fprintf(stderr, "cpyHostToDevice error at %s:%d: batchIndex out of range\n", __FILE__, __LINE__);
        std::exit(EXIT_FAILURE);
    }

    size_t dataSize = ((batchIndex == lastBatchIndex_) ? lastBatchSize_ : batchSize_) * sizeof(scalar);

    for (label i=0; i < systemSize_; i++)
    {
        CUDA_CHECK(cudaMemcpy(deviceRes_->vectors + i * batchSize_, hostRes_->vectors[i] + batchIndex * batchSize_, dataSize, cudaMemcpyHostToDevice));
    }

    for (label i=0; i < parameterSize_; i++)
    {
        CUDA_CHECK(cudaMemcpy(deviceRes_->parameters + i * batchSize_, hostRes_->parameters[i] + batchIndex * batchSize_, dataSize, cudaMemcpyHostToDevice));
    }
}

template<class HostResourcesType, class DeviceResourcesType>
void Operator<HostResourcesType, DeviceResourcesType>::cpyDeviceToHost(label batchIndex)
{
    if (batchIndex < 0 || batchIndex > lastBatchIndex_)
    {
        fprintf(stderr, "cpyDeviceToHost error at %s:%d: batchIndex out of range\n", __FILE__, __LINE__);
        std::exit(EXIT_FAILURE);
    }

    size_t dataSize = ((batchIndex == lastBatchIndex_) ? lastBatchSize_ : batchSize_) * sizeof(scalar);

    for (label i=0; i < systemSize_; i++)
    {
        CUDA_CHECK(cudaMemcpy(hostRes_->vectors[i] + batchIndex * batchSize_, deviceRes_->vectors + i * batchSize_, dataSize, cudaMemcpyDeviceToHost));
    }

    for (label i=0; i < parameterSize_; i++)
    {
        CUDA_CHECK(cudaMemcpy(hostRes_->parameters[i] + batchIndex * batchSize_, deviceRes_->parameters + i * batchSize_, dataSize, cudaMemcpyDeviceToHost));
    }
}

template<class HostResourcesType, class DeviceResourcesType>
label Operator<HostResourcesType, DeviceResourcesType>::getRealBatchSize(label batchIndex)
{
    if (batchIndex < 0 || batchIndex > lastBatchIndex_)
    {
        fprintf(stderr, "getRealBatchSize error at %s:%d: batchIndex out of range\n", __FILE__, __LINE__);
        std::exit(EXIT_FAILURE);
    }

    return (batchIndex == lastBatchIndex_) ? lastBatchSize_ : batchSize_;
}
}