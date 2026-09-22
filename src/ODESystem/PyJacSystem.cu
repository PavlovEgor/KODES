#include "PyJacSystem.cuh"

// The four host statics, all of them the shared ones. This file is compiled by
// nvcc and is where the vtable of a host side PyJacSystem is emitted; no caller
// ever has to emit one of its own.
KODES_DEFINE_DEVICE_OBJECT(kodes::PyJacSystem)

__host__ void
kodes::PyJacSystem::allocate()
{
    // pyJac allocates its device arrays onto a host struct and then copies the
    // struct to the device, which is the same two-step this factory does with
    // the object itself: the host half is what has to survive until
    // deallocate(), so it is a member and not a local.
    hostMemory_ = (mechanism_memory*)malloc(sizeof(mechanism_memory));

    if (!hostMemory_)
    {
        fprintf
        (
            stderr,
            "PyJacSystem::allocate error at %s:%d: out of host memory\n",
            __FILE__, __LINE__
        );
        std::exit(EXIT_FAILURE);
    }

    deviceMemory_ = nullptr;

    // Padded to the resident threads, not to the batch: every array in there
    // is one slot per thread, addressed with the same stride as the resources'
    // scratch.
    initialize_gpu_memory(scratchSize_, &hostMemory_, &deviceMemory_);
}

__host__ void
kodes::PyJacSystem::deallocate()
{
    if (hostMemory_)
    {
        free_gpu_memory(&hostMemory_, &deviceMemory_);

        free(hostMemory_);

        hostMemory_ = nullptr;
        deviceMemory_ = nullptr;
    }
}

__device__
void kodes::PyJacSystem::derivatives(const scalar t, const scalar pressure, const scalar* y, scalar* dy) const
{
    dydt(t, pressure, y, dy, deviceMemory_);
}

__device__
void kodes::PyJacSystem::jacobian(const scalar t, const scalar pressure, const scalar* y, scalar* dfdy) const
{
    eval_jacob(t, pressure, y, dfdy, deviceMemory_);
}
