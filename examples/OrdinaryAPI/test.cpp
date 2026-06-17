

int main(){

    kodes::data data;

    kodes::ODESystem ode;

    kodes::config config;

    kodes::integrator solver(ode, config);

    data.manage();

    solver.solve();



    return 0;
}