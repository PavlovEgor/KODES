#include "Resources.cuh"
#include "StepState.cuh"

namespace kodes 
{

class DeviceResources 
    :
    public Resources,
    public StepState
{
public:

    scalar*        vectors;
    scalar*        parameters;

    __device__
    DeviceResources(const label batchSize, const label systemSize, const label parameterSize) 
        : Resources(batchSize, systemSize, parameterSize), StepState(batchSize) {}

    __device__ __host__
    ~DeviceResources() = default;

    __device__ static void* operator new(size_t size, void* ptr) {
        return ptr;
    }
    
    __host__ static DeviceResources* 
    create(const label batchSize, const label systemSize, const label parameterSize);

    __host__ static void
    destroy(DeviceResources* devRes);

    __host__ __device__ void 
    printVectori(const label i) const;

    __device__ __host__ label batchSize() { return this->numOfSystems(); }
};

}

