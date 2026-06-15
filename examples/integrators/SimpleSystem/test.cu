
#include "seulex.cuh"
#include "euler.cuh"
#include "integrator.cuh"
#include <iostream>
#include "kodes_config.cuh"
#include "SimpleSystem.cuh"

int main(){

    std::string json_path = "config.json";

    kodes::Config config(json_path);

    kodes::SimpleSystem ode;

    kodes::seulex solver(ode, config);

    double xEnd(1);

    kodes::integrator::stepState ss(xEnd);

    double x = 0;
    std::vector<double> y = ode.getInitialConditions();

    do {
        solver.solve(x, y, ss);
    } while (x < xEnd);

    std::vector<double> yG = ode.getGroundSolution(x);


    for (const auto& element : y) {
        std::cout << element << " ";
    } std::cout << std::endl;

    for (const auto& element : yG) {
        std::cout << element << " ";
    } std::cout << std::endl;

    return 0;
}
