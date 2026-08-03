#include "seulex3.cuh"


int main(){

    label ensembleSize = 1024;

    kodes::HostResources            host_res(ensembleSize, NSP, 1);

    set_same_initial_conditions(host_res.ensembleSize(), host_res.vectors, host_res.parameters);

    host_res.printVectori(0);

    mechanism_memory *h_mem = (mechanism_memory*)malloc(sizeof(mechanism_memory));
    mechanism_memory *d_mem = nullptr;

    label batchSize = 1024;
    label numOfBatches = (ensembleSize + batchSize - 1) / batchSize;

    initialize_gpu_memory(batchSize, &h_mem, &d_mem);

    kodes::SeulexDeviceResources   host_res_dev(batchSize, host_res.systemSize(), host_res.parameterSize());

    kodes::SeulexDeviceResources*   res_prt = kodes::SeulexDeviceResources::create(batchSize, host_res.systemSize(), host_res.parameterSize(), &host_res_dev);

    kodes::pyJacSystem* ode_prt = kodes::pyJacSystem::createGPU(d_mem);

    kodes::Operator<kodes::HostResources, kodes::SeulexDeviceResources> op(&host_res, &host_res_dev);

    kodes::IntegratorControls controls(1e-10, 1e-1, 10000);

    kodes::Seulex<kodes::pyJacSystem> solver(ode_prt, res_prt, batchSize, controls);

    scalar xEnd = 10.0;

    solver.resetDeltaTMin();

    for (label i=0; i < numOfBatches; i++)
    {
        op.cpyHostToDevice(i);
        solver.solve(xEnd, op.getRealBatchSize(i));
        op.cpyDeviceToHost(i);
    }

    scalar deltaTMin = solver.deltaTMin();

    host_res.printVectori(0);

    printf("min deltaTTry over %d systems : %0.16e \n", ensembleSize, deltaTMin);

    kodes::pyJacSystem::destroyGPU(ode_prt);
    kodes::SeulexDeviceResources::destroy(res_prt, &host_res_dev);

    return 0;
}
