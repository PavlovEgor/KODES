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
#include "Settings.cuh"

#include "PyJacSystem.cuh"

// whichever mechanism this target was built against, see CMakeLists.txt
#include "gpu_memory.cuh"
#include "mechanism.cuh"

#endif
