#include "Resources.cuh"


namespace kodes 
{

class DeviceResources 
    :
    public Resources
{
public:

    scalar*        vectors;
    scalar*        parameters;

    __device__
    DeviceResources(const label numOfSystems, const label sizeOfSystem, const label numOfParameters) 
        : Resources(numOfSystems, sizeOfSystem, numOfParameters) {}

    __device__ __host__
    ~DeviceResources() = default;

    __device__ static void* operator new(size_t size, void* ptr) {
        return ptr;
    }
    
    __host__ static DeviceResources* 
    create(const label numOfSystems, const label sizeOfSystem, const label numOfParameters);

    __host__ static void
    destroy(DeviceResources* devRes);

    __host__ __device__ void 
    printVectori(const label i) const;
};

}

