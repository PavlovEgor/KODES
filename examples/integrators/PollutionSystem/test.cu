
#include "seulex.cuh"
#include <iostream>
#include "kodes_config.cuh"
#include "PollutionSystem.cuh"

int main(){

    std::string json_path = "config.json";

    kodes::Config config(json_path);

    kodes::PollutionSystem ode;

    kodes::seulex solver(ode, config);

    kodes::integrator::stepState ss(60.0);

    double x = 0;
    std::vector<double> y = ode.getInitialConditions();
    std::vector<double> yG = ode.getGroundSolution();

    solver.solve(x, y, ss);

    for (const auto& element : y) {
        std::cout << element << " ";
    } std::cout << std::endl;

    for (const auto& element : yG) {
        std::cout << element << " ";
    } std::cout << std::endl;

    return 0;
}
