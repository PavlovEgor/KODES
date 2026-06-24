#include "SeulexDeviceResources.cuh"

__device__ __host__
kodes::SeulexDeviceResources::SeulexDeviceResources(const label numOfSystems, const label sizeOfSystem)
:
    DeviceResources(numOfSystems, sizeOfSystem)
{
    cudaMalloc(&resources_scalar, 
        (
        12 * sizeOfSystem_ +                    // table_
        sizeOfSystem_ +                             // dfdx_
        sizeOfSystem_ * sizeOfSystem_ +             // dfdy_
        sizeOfSystem_ * sizeOfSystem_ +             // a_
        sizeOfSystem_ +                             // dxOpt_
        sizeOfSystem_ +                             // temp_
        sizeOfSystem_ +                             // y0_
        sizeOfSystem_ +                             // ySequence_
        sizeOfSystem_ +                             // scale_
        sizeOfSystem_ +                             // dy_
        sizeOfSystem_ +                             // yTemp_
        sizeOfSystem_ +                             // dydx_
        sizeOfSystem_                               // y
    ) * numOfSystems_*sizeof(scalar));

    cudaMalloc(&resources_label, sizeOfSystem_ * numOfSystems_*sizeof(label));

    cudaMalloc(&data_device, sizeOfSystem_ * numOfSystems_ * sizeof(scalar));
}

__device__ __host__
kodes::SeulexDeviceResources::~SeulexDeviceResources()
{
    cudaFree(data_device);
    cudaFree(resources_scalar);
    cudaFree(resources_label);
}

__host__ void
kodes::SeulexDeviceResources::cpyHostToDevice(scalar** in_data_host)
{
    for (label i=0; i < sizeOfSystem_; i++)
    {
        cudaMemcpy(data_device + i * numOfSystems_, in_data_host[i], numOfSystems_ * sizeof(scalar), cudaMemcpyHostToDevice);
    }
}

__host__ void
kodes::SeulexDeviceResources::cpyDeviceToHost(scalar** out_data_host) const
{
    for (label i=0; i < sizeOfSystem_; i++)
    {
        cudaMemcpy(out_data_host[i], data_device + i * numOfSystems_, numOfSystems_ * sizeof(scalar), cudaMemcpyDeviceToHost);
    }
}