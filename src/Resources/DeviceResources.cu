#include "basic_types.cuh"
#include "DeviceResources.cuh"

namespace kodes 
{

template<typename T>
T* DeviceResources::getResources() {
    static_assert(sizeof(T) == 0, "Unsupported type");
    return nullptr;
}

template<>
scalar* DeviceResources::getResources<scalar>() {
    return this->resources_scalar;
}

template<>
label* DeviceResources::getResources<label>() {
    return this->resources_label;
}

}