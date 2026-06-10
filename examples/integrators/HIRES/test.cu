
#include "seulex.cuh"
#include <iostream>
#include "kodes_config.cuh"
#include "HIRESSystem.cuh"

int main(){

    std::string json_path = "config.json";

    kodes::Config config(json_path);

    kodes::HIRESSystem ode;

    kodes::seulex solver(ode, config);

    kodes::integrator::stepState ss(321.8122);

    double x = 0;
    std::vector<double> y = ode.getInitialConditions();

    solver.solve(x, y, ss);

    return 0;
}
