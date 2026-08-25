#include "balancer_table.cuh"

#include "TemperatureBalancer.cuh"
#include "RHSNormBalancer.cuh"
#include "StiffnessBalancer.cuh"

namespace kodes
{

// Adding a balancer is a subclass with a key(), the one line of
// KODES_DEFINE_DEVICE_OBJECT in its .cu, and one line here.
static const TypeEntry<Balancer> table[] =
{
    typeEntry<Balancer, TemperatureBalancer>("temperature"),
    typeEntry<Balancer, RHSNormBalancer>("rhsNorm"),
    typeEntry<Balancer, StiffnessBalancer>("stiffness")
};

static const label tableSize = label(sizeof(table)/sizeof(table[0]));

}

__host__ const kodes::TypeEntry<kodes::Balancer>* kodes::balancerTable()
{
    return kodes::table;
}

__host__ label kodes::balancerTableSize()
{
    return kodes::tableSize;
}

__host__ const kodes::TypeEntry<kodes::Balancer>*
kodes::balancerType(const char* name)
{
    if (!name || !*name || strcmp(name, kNoBalancer) == 0)
    {
        return nullptr;
    }

    return findType<Balancer>(table, tableSize, name, "balancer");
}

__host__ size_t kodes::balancerStateBytesPerSystem
(
    const char* name,
    const label systemSize,
    const label parameterSize
)
{
    const TypeEntry<Balancer>* type = balancerType(name);

    return type ? type->stateBytesPerSystem(systemSize, parameterSize) : 0;
}

__host__ size_t kodes::balancerScratchBytesPerThread
(
    const char* name,
    const label systemSize,
    const label parameterSize
)
{
    const TypeEntry<Balancer>* type = balancerType(name);

    return type ? type->scratchBytesPerThread(systemSize, parameterSize) : 0;
}

__host__ kodes::Handle<kodes::Balancer> kodes::newBalancer
(
    const char* name,
    const label batchSize,
    const label scratchSize,
    const label systemSize,
    const label parameterSize
)
{
    const TypeEntry<Balancer>* type = balancerType(name);

    if (!type)
    {
        return Handle<Balancer>();
    }

    return Handle<Balancer>
    (
        type, batchSize, scratchSize, systemSize, parameterSize
    );
}
