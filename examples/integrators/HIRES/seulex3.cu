#include "seulex3.cuh"


int main(){

    label numOfSystems = 1 << 5;

    kodes::HostResources            host_res(numOfSystems, 8, 0);

    init(&host_res);

    host_res.printVectori(0);

    kodes::HIRESSystem* ode_prt = kodes::HIRESSystem::createGPU(numOfSystems);

    kodes::SeulexDeviceResources   host_res_dev(host_res.numOfSystems(), host_res.sizeOfSystem(), host_res.numOfParameters());

    kodes::SeulexDeviceResources*   res_prt = kodes::SeulexDeviceResources::create(numOfSystems, host_res.sizeOfSystem(), 1, &host_res_dev);

    kodes::Operator<kodes::HostResources, kodes::SeulexDeviceResources> op(&host_res, &host_res_dev);

    op.cpyHostToDevice();

    scalar xEnd = 321.8122;
    stepState step(xEnd);

    kodes::Seulex<kodes::HIRESSystem> solver(ode_prt, res_prt, step, host_res.numOfSystems());

    solver.solve();
    
    op.cpyDeviceToHost();

    host_res.printVectori(0);

    kodes::HIRESSystem::destroyGPU(ode_prt);
    kodes::SeulexDeviceResources::destroy(res_prt, &host_res_dev);

    return 0;
}

void init(kodes::HostResources* host_res)
{
    for (label i=0; i < host_res -> sizeOfSystem(); ++i)
    {
        host_res -> vectors[i] = (scalar*)malloc(host_res -> numOfSystems() * sizeof(scalar));
        for (label j=0; j<host_res -> numOfSystems(); ++j)
        {
            host_res -> vectors[i][j] = 0;
        }
    }
    for (label j=0; j<host_res -> numOfSystems(); ++j)
    {
        host_res -> vectors[0][j] = 1.0;
        host_res -> vectors[7][j] = 0.0057;
    }
}
