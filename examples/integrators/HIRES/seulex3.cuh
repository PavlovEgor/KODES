#pragma once

#include "basic_linalg.cuh"
#include "SeulexDeviceResources.cuh"
#include "HIRESSystem.cuh"
#include "HostResources.cuh"
#include "Operator.cuh"
#include "Seulex.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda/cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono> 

void init(kodes::HostResources* vectors);
