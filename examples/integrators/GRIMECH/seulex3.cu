#include "seulex3.cuh"


int main(){

    label numOfSystems = 1 << 5;

    kodes::HostResources            host_res(numOfSystems, NSP, 1);

    set_same_initial_conditions(host_res.numOfSystems(), host_res.vectors, host_res.parameters);

    host_res.printVectori(0);

    mechanism_memory *h_mem = (mechanism_memory*)malloc(sizeof(mechanism_memory));
    mechanism_memory *d_mem = nullptr;

    initialize_gpu_memory(host_res.numOfSystems(), &h_mem, &d_mem);

    kodes::SeulexDeviceResources   host_res_dev(host_res.numOfSystems(), host_res.sizeOfSystem(), host_res.numOfParameters());

    kodes::SeulexDeviceResources*   res_prt = kodes::SeulexDeviceResources::create(numOfSystems, host_res.sizeOfSystem(), 1, &host_res_dev);

    kodes::GRIMESHSystem* ode_prt = kodes::GRIMESHSystem::createGPU(d_mem);

    kodes::Operator<kodes::HostResources, kodes::SeulexDeviceResources> op(&host_res, &host_res_dev);

    op.cpyHostToDevice();

    scalar xEnd = 1.0e-3;
    stepState step(xEnd);

    kodes::Seulex<kodes::GRIMESHSystem> solver(ode_prt, res_prt, step, host_res.numOfSystems());

    solver.solve();
    
    op.cpyDeviceToHost();

    host_res.printVectori(0);

    kodes::GRIMESHSystem::destroyGPU(ode_prt);
    kodes::SeulexDeviceResources::destroy(res_prt, &host_res_dev);

    return 0;
}
