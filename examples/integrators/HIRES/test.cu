
#include "seulex.cuh"
#include "euler.cuh"
#include <iostream>
#include "kodes_config.cuh"
#include "HIRESSystem.cuh"
#include <chrono>


int main(){

    auto start_total = std::chrono::high_resolution_clock::now();

    int numOfSystems = 1 << 20;

    auto start_config = std::chrono::high_resolution_clock::now();
    std::string json_path = "config.json";
    kodes::Config config(json_path);
    auto end_config = std::chrono::high_resolution_clock::now();
    auto duration_config = std::chrono::duration_cast<std::chrono::microseconds>(end_config - start_config);
    std::cout << "Время загрузки конфигурации: " << duration_config.count() << " мкс" << std::endl;

    auto start_system = std::chrono::high_resolution_clock::now();
    kodes::HIRESSystem ode;
    kodes::euler solver(ode, config);
    auto end_system = std::chrono::high_resolution_clock::now();
    auto duration_system = std::chrono::duration_cast<std::chrono::microseconds>(end_system - start_system);
    std::cout << "Время создания системы: " << duration_system.count() << " мкс" << std::endl;

    double xEnd(321.8122);
    kodes::integrator::stepState ss(xEnd);
    double x = 0;
    std::vector<double> y = ode.getInitialConditions();

    auto start_solve = std::chrono::high_resolution_clock::now();

    for (int i=0; i<numOfSystems; ++i)
    {
    kodes::integrator::stepState ss(xEnd);
    y = ode.getInitialConditions();
    x = 0;    
    do {
        solver.solve(x, y, ss);
    } while (x < xEnd);
    }

    auto end_solve = std::chrono::high_resolution_clock::now();
    auto duration_solve = std::chrono::duration_cast<std::chrono::milliseconds>(end_solve - start_solve);
    std::cout << "Время решения (основная часть): " << duration_solve.count() << " мс" << std::endl;
    std::vector<double> yG = ode.getGroundSolution();


    for (const auto& element : y) {
        std::cout << element << " ";
    } std::cout << std::endl;

    for (const auto& element : yG) {
        std::cout << element << " ";
    } std::cout << std::endl;

    auto end_total = std::chrono::high_resolution_clock::now();
    auto duration_total = std::chrono::duration_cast<std::chrono::milliseconds>(end_total - start_total);
    std::cout << "\nОбщее время выполнения: " << duration_total.count() << " мс" << std::endl;

    return 0;
}
