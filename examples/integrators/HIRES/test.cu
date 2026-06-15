
#include "seulex.cuh"
#include <iostream>
#include "kodes_config.cuh"
#include "HIRESSystem.cuh"

int main(){

    std::string json_path = "config.json";

    kodes::Config config(json_path);

    kodes::HIRESSystem ode;

    kodes::seulex solver(ode, config);

    double xEnd(321.8122);

    kodes::integrator::stepState ss(xEnd);

    double x = 0;
    std::vector<double> y = ode.getInitialConditions();

    do {
        solver.solve(x, y, ss);
    } while (x < xEnd);

    std::vector<double> yG = ode.getGroundSolution();


    for (const auto& element : y) {
        std::cout << element << " ";
    } std::cout << std::endl;

    for (const auto& element : yG) {
        std::cout << element << " ";
    } std::cout << std::endl;

    return 0;
}
