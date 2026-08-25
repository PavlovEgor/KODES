#include "method_table.cuh"

#include "Seulex.cuh"
#include "SeulexDeviceResources.cuh"

#include "Euler.cuh"
#include "EulerDeviceResources.cuh"

namespace kodes
{

// Adding a method is a subclass with a step(), the one line of
// KODES_DEFINE_DEVICE_OBJECT in its .cu, and one line here naming the resources
// that hold its scratch.
static const MethodType table[] =
{
    {
        "seulex",
        typeEntry<IntegrationMethod, Seulex>("seulex"),
        typeEntry<DeviceResources, SeulexDeviceResources>("seulexResources")
    },
    {
        "euler",
        typeEntry<IntegrationMethod, Euler>("euler"),
        typeEntry<DeviceResources, EulerDeviceResources>("eulerResources")
    }
};

static const label tableSize = label(sizeof(table)/sizeof(table[0]));

}

__host__ const kodes::MethodType* kodes::methodTable()
{
    return kodes::table;
}

__host__ label kodes::methodTableSize()
{
    return kodes::tableSize;
}

__host__ const kodes::MethodType* kodes::methodType(const char* name)
{
    if (!name || !*name)
    {
        fprintf(stderr, "kodes::methodType error at %s:%d: no integration method named\n", __FILE__, __LINE__);
        std::exit(EXIT_FAILURE);
    }

    for (label i = 0; i < tableSize; ++i)
    {
        if (strcmp(table[i].name, name) == 0)
        {
            return table + i;
        }
    }

    fprintf(stderr, "kodes::methodType error at %s:%d: unknown integration method \"%s\", known are", __FILE__, __LINE__, name);
    for (label i = 0; i < tableSize; ++i)
    {
        fprintf(stderr, " \"%s\"", table[i].name);
    }
    fprintf(stderr, "\n");
    std::exit(EXIT_FAILURE);

    return nullptr;
}

__host__ size_t kodes::methodScratchBytesPerThread
(
    const char* name,
    const label systemSize,
    const label parameterSize
)
{
    const MethodType* type = methodType(name);

    return type->method.scratchBytesPerThread(systemSize, parameterSize)
         + type->resources.scratchBytesPerThread(systemSize, parameterSize);
}

__host__ size_t kodes::methodStateBytesPerSystem
(
    const char* name,
    const label systemSize,
    const label parameterSize
)
{
    const MethodType* type = methodType(name);

    return type->method.stateBytesPerSystem(systemSize, parameterSize)
         + type->resources.stateBytesPerSystem(systemSize, parameterSize);
}

__host__ kodes::Handle<kodes::IntegrationMethod> kodes::newMethod
(
    const char* name,
    const label batchSize,
    const label scratchSize,
    const label systemSize,
    const label parameterSize
)
{
    return Handle<IntegrationMethod>
    (
        &methodType(name)->method, batchSize, scratchSize, systemSize, parameterSize
    );
}

__host__ kodes::Handle<kodes::DeviceResources> kodes::newResources
(
    const char* name,
    const label batchSize,
    const label scratchSize,
    const label systemSize,
    const label parameterSize
)
{
    return Handle<DeviceResources>
    (
        &methodType(name)->resources, batchSize, scratchSize, systemSize, parameterSize
    );
}
