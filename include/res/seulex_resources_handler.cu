#include "seulex_resources_handler.cuh"


kodes::SeulexDeviceResources::SeulexDeviceResources(const label numOfSystems, const label sizeOfSystem)
:
    numOfSystems_(numOfSystems),
    sizeOfSystem_(sizeOfSystem)
{
    cudaMalloc(&resouces_scalar, 
        (
        kMaxx_ * sizeOfSystem_ +                    // table_
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

    cudaMalloc(&resouces_label, sizeOfSystem_ * numOfSystems_*sizeof(label));

    cudaMalloc(&data_device, sizeOfSystem_ * numOfSystems_ * sizeof(scalar));
}

kodes::SeulexDeviceResources::~SeulexDeviceResources()
{
    cudaFree(dev_data);
    cudaFree(resouces_scalar);
    cudaFree(resouces_label);
}

void
kodes::SeulexDeviceResources::cpyHostToDevice(scalar** in_data_host)
{
    for (label i=0; i < sizeOfSystem_; i++)
    {
        cudaMemcpy(data_device + i * numOfSystems_, in_data_host[i], numOfSystems_ * sizeof(scalar), cudaMemcpyHostToDevice);
    }
}

void
kodes::SeulexDeviceResources::cpyDeviceToHost(const scalar** out_data_host) const
{
    for (label i=0; i < sizeOfSystem_; i++)
    {
        cudaMemcpy(out_data_host[i], dev_data + i * numOfSystems_, numOfSystems_ * sizeof(scalar), cudaMemcpyDeviceToHost);
    }
}