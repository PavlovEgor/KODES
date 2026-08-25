
#include <iostream>
#include "Config.cuh"


int main(){

    std::string json_path = "config.json";

    kodes::Config config(json_path);

    std::cout << config.getInt("maxSteps", 1) << std::endl;
    
    return 0;
}
