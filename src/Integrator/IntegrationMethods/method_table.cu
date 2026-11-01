#include "method_table.cuh"

#include "Seulex/Seulex.cuh"
#include "Euler/Euler.cuh"
#include "Rosenbrock34/Rosenbrock34.cuh"

namespace kodes
{

// Adding a method is a subclass with an advance() - or, taking trial steps, an
// AdaptiveIntegrationMethod with a trialStep() - the one line of
// KODES_DEFINE_DEVICE_OBJECT in its .cu, and one line here.
static const TypeEntry<IntegrationMethod> table[] =
{
    typeEntry<IntegrationMethod, Seulex>("seulex"),
    typeEntry<IntegrationMethod, Euler>("euler"),
    typeEntry<IntegrationMethod, Rosenbrock34>("rosenbrock34")
};

static const label tableSize = label(sizeof(table)/sizeof(table[0]));

}

__host__ const kodes::TypeEntry<kodes::IntegrationMethod>* kodes::methodTable()
{
    return kodes::table;
}

__host__ kodes::label kodes::methodTableSize()
{
    return kodes::tableSize;
}

__host__ const kodes::TypeEntry<kodes::IntegrationMethod>*
kodes::methodType(const char* name)
{
    if (!name || !*name)
    {
        fprintf(stderr, "kodes::methodType error at %s:%d: no integration method named\n", __FILE__, __LINE__);
        std::exit(EXIT_FAILURE);
    }

    return findType<IntegrationMethod>(table, tableSize, name, "integration method");
}

__host__ size_t kodes::methodScratchBytesPerThread
(
    const char* name,
    const label systemSize,
    const label parameterSize
)
{
    return methodType(name)->scratchBytesPerThread(systemSize, parameterSize);
}

__host__ size_t kodes::methodScratchBytesPerBlock
(
    const char* name,
    const label systemSize,
    const label parameterSize
)
{
    return methodType(name)->scratchBytesPerBlock(systemSize, parameterSize);
}

__host__ size_t kodes::methodStateBytesPerSystem
(
    const char* name,
    const label systemSize,
    const label parameterSize
)
{
    return methodType(name)->stateBytesPerSystem(systemSize, parameterSize);
}

__host__ kodes::Handle<kodes::IntegrationMethod> kodes::newMethod
(
    const char* name,
    const label batchSize,
    const label scratchSize,
    const label blocks,
    const label systemSize,
    const label parameterSize
)
{
    return Handle<IntegrationMethod>
    (
        methodType(name), batchSize, scratchSize, blocks, systemSize, parameterSize
    );
}
