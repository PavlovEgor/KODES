#include "TemperatureBalancer.cuh"

// The four host statics, all of them the shared ones. This file is compiled by
// nvcc and is where the vtable of a host side TemperatureBalancer is emitted;
// no caller ever has to emit one of its own.
KODES_DEFINE_DEVICE_OBJECT(kodes::TemperatureBalancer)
