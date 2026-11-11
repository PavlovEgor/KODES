#ifndef KODES_HOST_RESOURCES
#define KODES_HOST_RESOURCES

#pragma once

#include <array>

#include "Resources.cuh"
#include "system_dimensions.cuh"


namespace kodes 
{

class HostResources 
    :
    public Resources
{
public:
    // One pointer per component and per parameter, each to the caller's array
    // of that component across every system. The arrays stay the caller's;
    // only these lists of pointers are this object's, so a copy points at the
    // same arrays. As many as the library is built for, see
    // system_dimensions.cuh.
    std::array<scalar*, kSystemSize> vectors;
    std::array<scalar*, kParameterSize> parameters;

    HostResources(const label ensembleSize, const label systemSize, const label parameterSize);

    HostResources& operator=(const HostResources& other);

    void printVectori(const label i) const;
    
    void printParameteri(const label i) const;

    void setVector(const label i, scalar* vector) { this->vectors[i] = vector; }

    void setParameter(const label i, scalar* parameter) { this->parameters[i] = parameter; }
};

}


#endif
