#include "seulex3.cuh"


int main(){

    label numOfSystems = 2050;

    kodes::HostResources            host_res(numOfSystems, NSP, 1);

    set_same_initial_conditions(host_res.numOfSystems(), host_res.vectors, host_res.parameters);

    host_res.printVectori(0);

    mechanism_memory *h_mem = (mechanism_memory*)malloc(sizeof(mechanism_memory));
    mechanism_memory *d_mem = nullptr;

    label batchSize = 1024;
    label numOfBatches = (numOfSystems + batchSize - 1) / batchSize;

    initialize_gpu_memory(batchSize, &h_mem, &d_mem);

    kodes::SeulexDeviceResources   host_res_dev(batchSize, host_res.systemSize(), host_res.parameterSize());

    kodes::SeulexDeviceResources*   res_prt = kodes::SeulexDeviceResources::create(batchSize, host_res.systemSize(), host_res.parameterSize(), &host_res_dev);

    kodes::pyJacSystem* ode_prt = kodes::pyJacSystem::createGPU(d_mem);

    kodes::Operator<kodes::HostResources, kodes::SeulexDeviceResources> op(&host_res, &host_res_dev);

    kodes::Seulex<kodes::pyJacSystem> solver(ode_prt, res_prt, batchSize);

    scalar xEnd = 10.0;
    kodes::stepState step(xEnd);

    for (label i=0; i < numOfBatches; i++)
    {
        op.cpyHostToDevice(i);
        solver.solve(step, op.getRealBatchSize(i));
        op.cpyDeviceToHost(i); 
    }

    host_res.printVectori(0);

    kodes::pyJacSystem::destroyGPU(ode_prt);
    kodes::SeulexDeviceResources::destroy(res_prt, &host_res_dev);

    return 0;
}
