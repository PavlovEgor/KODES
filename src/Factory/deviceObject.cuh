#ifndef KODES_DEVICE_OBJECT
#define KODES_DEVICE_OBJECT

#pragma once

#include <cuda_runtime.h>

#include "basicTypes.cuh"

// Construction and destruction of a *device object*: a class whose virtual
// functions are called from inside a kernel, and which therefore has to live in
// device memory with a device vtable.
//
// The same four steps for every such class in the library - the balancer, the
// resources, the integration method - so they are written once here and every
// concrete class hands out four one-line forwards into them.
//
// The contract a class has to keep to be built this way:
//
//   1. a `__device__ __host__` constructor taking exactly
//      (batchSize, scratchSize, systemSize, parameterSize), which sets value
//      members only and never touches a buffer pointer - see create() for why;
//   2. `__host__ void allocate()` / `deallocate()`, which cudaMalloc/cudaFree
//      the buffers into the object's own pointers. They are *not* virtual: the
//      templates below are instantiated with the concrete class, so the
//      concrete pair is the one that runs. A subclass calls its base's first;
//   3. `__device__ static void* operator new(size_t, void*)`, so that the
//      constructor kernel can placement-new into device memory.
//
// A class with nothing to allocate inherits an empty allocate() from its base
// and is built by the same calls.

namespace kodes
{

template<class T>
__global__ void constructDeviceObjectKernel
(
    T* object,
    const label batchSize,
    const label scratchSize,
    const label systemSize,
    const label parameterSize
)
{
    new (object) T(batchSize, scratchSize, systemSize, parameterSize);
}

template<class T>
__global__ void destructDeviceObjectKernel(T* object)
{
    object->~T();
}

// The host side twin of the device object. It holds the same members, and it is
// where the device buffers are allocated before they are handed over.
//
// Held as a pointer and made here rather than declared by the caller, because a
// class with a device-only virtual can only have its *host* vtable emitted by a
// compiler that invents a host stub for one: nvcc does, `nvc++ -cuda` does not
// and stops with the virtual undefined in the vtable. A caller compiled by
// anything other than nvcc - an OpenFOAM chemistry model, say - must therefore
// let a .cu of this library construct it.
template<class T>
__host__ T* createStubObject
(
    const label batchSize,
    const label scratchSize,
    const label systemSize,
    const label parameterSize
)
{
    return ::new T(batchSize, scratchSize, systemSize, parameterSize);
}

template<class T>
__host__ void destroyStubObject(T* hostStub)
{
    ::delete hostStub;
}

// Allocate the buffers on the stub, hand them to a fresh device object, and
// give it a device vtable.
//
// The order matters: the stub is byte-copied over first so that the device
// object inherits the addresses just allocated, and only then constructed in
// place, since it is the constructor that writes the device vtable pointer.
// That is also why the constructor must leave every buffer pointer alone - it
// runs after the copy and would overwrite what the copy brought.
template<class T>
__host__ T* createDeviceObject
(
    const label batchSize,
    const label scratchSize,
    const label systemSize,
    const label parameterSize,
    T* hostStub
)
{
    if (!hostStub)
    {
        fprintf(stderr, "kodes::createDeviceObject error at %s:%d: hostStub is null\n", __FILE__, __LINE__);
        std::exit(EXIT_FAILURE);
    }

    if (batchSize <= 0 || scratchSize <= 0 || systemSize <= 0 || parameterSize < 0)
    {
        fprintf
        (
            stderr,
            "kodes::createDeviceObject error at %s:%d: batchSize %d, scratchSize %d "
            "and systemSize %d must be positive and parameterSize %d not negative\n",
            __FILE__, __LINE__, batchSize, scratchSize, systemSize, parameterSize
        );
        std::exit(EXIT_FAILURE);
    }

    T* devPtr;
    CUDA_CHECK(cudaMalloc(&devPtr, sizeof(T)));

    hostStub->allocate();

    CUDA_CHECK(cudaMemcpy(devPtr, hostStub, sizeof(T), cudaMemcpyHostToDevice));

    constructDeviceObjectKernel<T><<<1, 1>>>
    (
        devPtr, batchSize, scratchSize, systemSize, parameterSize
    );
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());

    return devPtr;
}

template<class T>
__host__ void destroyDeviceObject(T* devObject, T* hostStub)
{
    if (hostStub)
    {
        hostStub->deallocate();
    }

    if (devObject)
    {
        destructDeviceObjectKernel<T><<<1, 1>>>(devObject);
        CUDA_CHECK_LAST();
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaFree(devObject));
    }
}

}

// The four host statics every concrete device object hands out, in one line.
// Written in the .cu of the class - never in a header a caller includes, since
// these are what launch the construction kernels.
//
// A class that needs more than the four steps (uploading a table of constants,
// say) writes its create() out by hand and calls createDeviceObject() from it.
#define KODES_DEFINE_DEVICE_OBJECT(Class)                                      \
                                                                               \
__host__ Class* Class::create                                                  \
(                                                                              \
    const label batchSize,                                                     \
    const label scratchSize,                                                   \
    const label systemSize,                                                    \
    const label parameterSize,                                                 \
    Class* hostStub                                                            \
)                                                                              \
{                                                                              \
    return kodes::createDeviceObject<Class>                                    \
    (                                                                          \
        batchSize, scratchSize, systemSize, parameterSize, hostStub            \
    );                                                                         \
}                                                                              \
                                                                               \
__host__ void Class::destroy(Class* devObject, Class* hostStub)                \
{                                                                              \
    kodes::destroyDeviceObject<Class>(devObject, hostStub);                    \
}                                                                              \
                                                                               \
__host__ Class* Class::createStub                                              \
(                                                                              \
    const label batchSize,                                                     \
    const label scratchSize,                                                   \
    const label systemSize,                                                    \
    const label parameterSize                                                  \
)                                                                              \
{                                                                              \
    return kodes::createStubObject<Class>                                      \
    (                                                                          \
        batchSize, scratchSize, systemSize, parameterSize                      \
    );                                                                         \
}                                                                              \
                                                                               \
__host__ void Class::destroyStub(Class* hostStub)                              \
{                                                                              \
    kodes::destroyStubObject<Class>(hostStub);                                 \
}

// The declarations that go with it, in the class body.
#define KODES_DECLARE_DEVICE_OBJECT(Class)                                     \
                                                                               \
    __device__ static void* operator new(size_t, void* ptr) { return ptr; }    \
                                                                               \
    __host__ static Class* create                                              \
    (                                                                          \
        const label batchSize,                                                 \
        const label scratchSize,                                               \
        const label systemSize,                                                \
        const label parameterSize,                                             \
        Class* hostStub                                                        \
    );                                                                         \
                                                                               \
    __host__ static void destroy(Class* devObject, Class* hostStub);           \
                                                                               \
    __host__ static Class* createStub                                          \
    (                                                                          \
        const label batchSize,                                                 \
        const label scratchSize,                                               \
        const label systemSize,                                                \
        const label parameterSize                                              \
    );                                                                         \
                                                                               \
    __host__ static void destroyStub(Class* hostStub);

#endif
