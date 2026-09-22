#include "system_table.cuh"

#include "PyJacSystem/PyJacSystem.cuh"

namespace kodes
{

// Adding a system is a subclass with a derivatives() and a jacobian(), the one
// line of KODES_DEFINE_DEVICE_OBJECT in its .cu, and one line here.
//
// This is the only translation unit of the library that includes a concrete
// system, and therefore the only one that needs a pyJac mechanism on its
// include path - along with PyJacSystem.cu itself.
static const TypeEntry<ODESystem> table[] =
{
    typeEntry<ODESystem, PyJacSystem>(kDefaultSystem)
};

static const label tableSize = label(sizeof(table)/sizeof(table[0]));

}

__host__ const kodes::TypeEntry<kodes::ODESystem>* kodes::systemTable()
{
    return kodes::table;
}

__host__ label kodes::systemTableSize()
{
    return kodes::tableSize;
}

__host__ const kodes::TypeEntry<kodes::ODESystem>*
kodes::systemType(const char* name)
{
    if (!name || !*name)
    {
        fprintf(stderr, "kodes::systemType error at %s:%d: no ODE system named\n", __FILE__, __LINE__);
        std::exit(EXIT_FAILURE);
    }

    return findType<ODESystem>(table, tableSize, name, "ODE system");
}

__host__ size_t kodes::systemScratchBytesPerThread
(
    const char* name,
    const label systemSize,
    const label parameterSize
)
{
    return systemType(name)->scratchBytesPerThread(systemSize, parameterSize);
}

__host__ size_t kodes::systemScratchBytesPerBlock
(
    const char* name,
    const label systemSize,
    const label parameterSize
)
{
    return systemType(name)->scratchBytesPerBlock(systemSize, parameterSize);
}

__host__ size_t kodes::systemStateBytesPerSystem
(
    const char* name,
    const label systemSize,
    const label parameterSize
)
{
    return systemType(name)->stateBytesPerSystem(systemSize, parameterSize);
}

__host__ kodes::Handle<kodes::ODESystem> kodes::newSystem
(
    const char* name,
    const label batchSize,
    const label scratchSize,
    const label blocks,
    const label systemSize,
    const label parameterSize
)
{
    return Handle<ODESystem>
    (
        systemType(name), batchSize, scratchSize, blocks, systemSize, parameterSize
    );
}
