#ifndef KODES_HOST_RESOURCES
#define KODES_HOST_RESOURCES

#pragma once

#include "Resources.cuh"


namespace kodes 
{

class HostResources 
    :
    public Resources
{
public:
    scalar**        vectors;
    scalar**        parameters;

    HostResources(const label ensembleSize, const label systemSize, const label parameterSize);
    
    __device__ __host__
    ~HostResources();

    HostResources& operator=(const HostResources& other);

    void printVectori(const label i) const;
    
    void printParameteri(const label i) const;

    void setVector(const label i, scalar* vector) { this->vectors[i] = vector; }

    void setParameter(const label i, scalar* parameter) { this->parameters[i] = parameter; }
};

}


#endif
