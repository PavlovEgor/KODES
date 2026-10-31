#ifndef KODES_DEVICE_BUFFER
#define KODES_DEVICE_BUFFER

#pragma once

#include "basic_types.cuh"

namespace kodes
{

// The base of the views of device memory, which every buffer of the library
// is held and passed as: ThreadVector, ThreadMatrix and ThreadValue in
// LinearAlgebra/, BatchArray here, Seulex's ExtrapolationTable. It holds the
// buffer, and a view adds how its elements are laid out in it.
//
// A view owns nothing. Copying one, or passing it by value, gives another name
// for the same memory, as copying a pointer does. Assigning to one writes the
// elements, as assigning through a reference does: `y = yTemp` copies the
// vector, `y = yTemp + dy` sums into it, neither repoints y. A view there is
// nothing sensible to assign to - a matrix, the batch - cannot be assigned at
// all, so that nobody repoints one by accident either.
//
// A default constructed view is left alone, which is what the device objects
// need: their constructors run after the stub has been byte-copied to the
// device, and must not overwrite what the copy brought (see
// Factory/device_object.cuh). The views' allocate() and deallocate() are
// called on that stub, on the host.
template <typename T>
class DeviceBuffer
{
protected:

    T* data_;

    DeviceBuffer() = default;

    DeviceBuffer(const DeviceBuffer&) = default;

    // never repoints a view; a view that can be assigned writes elements
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    __host__ __device__ explicit DeviceBuffer(T* data)
        : data_(data)
    {}

    // `count` elements in all, for a view's allocate() to size
    __host__ void allocateElements(const size_t count)
    {
        CUDA_CHECK(cudaMalloc(&data_, count * sizeof(T)));
    }

public:

    // The bare buffer, for what takes one: pyJac in PyJacSystem
    __host__ __device__ T* data() const
    {
        return data_;
    }

    __host__ void deallocate()
    {
        CUDA_CHECK(cudaFree(data_));
    }
};

}

#endif
