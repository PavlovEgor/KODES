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

    HostResources(const label numOfSystems, const label sizeOfSystem, const label numOfParameters);
    
    __device__ __host__
    ~HostResources();

    HostResources& operator=(const HostResources& other);

    void printVectori(const label i) const;
    
    void printParameteri(const label i) const;

    void setVector(const label i, scalar* vector) { this->vectors[i] = vector; }

    void setParameter(const label i, scalar* parameter) { this->parameters[i] = parameter; }
};

}

