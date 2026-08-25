#include "PyJacSystem.cuh"


__global__ void 
constructPyJacSystemKernel(kodes::PyJacSystem* system, mechanism_memory *d_mem)
{
    new (system) kodes::PyJacSystem(d_mem);
}

__global__ void 
destructPyJacSystemKernel(kodes::PyJacSystem* system) {
    system->~PyJacSystem();
}

__host__  kodes::PyJacSystem* 
kodes::PyJacSystem::create(mechanism_memory *d_mem) {
    if (!d_mem)
    {
        fprintf(stderr, "PyJacSystem::create error at %s:%d: d_mem is null\n", __FILE__, __LINE__);
        std::exit(EXIT_FAILURE);
    }

    PyJacSystem* ptr;
    CUDA_CHECK(cudaMalloc(&ptr, sizeof(PyJacSystem)));
    constructPyJacSystemKernel<<<1, 1>>>(ptr, d_mem);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());
    return ptr;
}

__host__  void
kodes::PyJacSystem::destroy(kodes::PyJacSystem* system) {
    if (system) {
        destructPyJacSystemKernel<<<1, 1>>>(system);
        CUDA_CHECK_LAST();
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(system));
    }
}

__device__
void kodes::PyJacSystem::derivatives(const scalar t, const scalar pressure, const scalar* y, scalar* dy) const
{
    dydt(t, pressure, y, dy, device_memory);
}

__device__
void kodes::PyJacSystem::jacobian(const scalar t, const scalar pressure, const scalar* y, scalar* dfdt, scalar* dfdy) const
{
    eval_jacob(t, pressure, y, dfdy, device_memory);
}
