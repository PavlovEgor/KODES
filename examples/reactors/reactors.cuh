#ifndef KODES_REACTORS_EXAMPLE
#define KODES_REACTORS_EXAMPLE

#pragma once

#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>

#include "basic_types.cuh"
#include "HostResources.cuh"
#include "DeviceResources.cuh"
#include "Operator.cuh"

#include "Integrator.cuh"
#include "IntegrationMethod.cuh"
#include "method_table.cuh"
#include "balancer_table.cuh"
#include "system_table.cuh"
#include "Settings.cuh"

// whichever mechanism this target was built against, see CMakeLists.txt. Only
// NSP and the names printed below come from here: the system itself is built
// through system_table.cuh, which names no mechanism and includes none of
// pyJac's headers.
#include "mechanism.cuh"

#endif
