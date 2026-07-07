// pyjac_probe.cu
//
// Точечная проверка: вызывает те же сгенерированные pyJac функции (dydt,
// eval_jacob), что и kodes::GRIMESHSystem::derivatives/jacobian, но для ОДНОГО
// состояния (T, P, состав), без интегратора Seulex. Печатает все промежуточные
// величины (конц-и, скорости реакций, net production rates, dy/dt), чтобы можно
// было построчно сравнить их с cantera_reference.py для того же состояния.
//
// Сборка: см. CMakeLists.txt в этой же папке.
// Запуск:  ./pyjac_probe [state.txt] > pyjac_output.txt
//
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cuda_runtime.h>

#include "header.cuh"
#include "mechanism.cuh"
#include "mass_mole.cuh"
#include "chem_utils.cuh"
#include "dydt.cuh"
#include "jacob.cuh"
#include "gpu_memory.cuh"

// Порядок видов должен совпадать с комментарием "Species Indexes" в
// mechanism.cuh - если у вас другой .cti/.yaml или другая версия pyJac,
// проверьте этот список перед сравнением с Cantera.
static const char* SPECIES_NAMES[NSP] = {
    "H2","H","O","O2","OH","H2O","HO2","H2O2","C","CH","CH2","CH2(S)","CH3","CH4",
    "CO","CO2","HCO","CH2O","CH2OH","CH3O","CH3OH","C2H","C2H2","C2H3","C2H4","C2H5",
    "C2H6","HCCO","CH2CO","HCCOH","N","NH","NH2","NH3","NNH","NO","NO2","N2O","HNO",
    "CN","HCN","H2CN","HCNN","HCNO","HOCN","HNCO","NCO","AR","C3H7","C3H8","CH2CHO",
    "CH3CHO","N2"
};

#define CUDA_CHECK(x) do { \
    cudaError_t _e = (x); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
        std::exit(1); \
    } \
} while (0)

// ВАЖНО: раньше здесь ещё вызывался eval_jacob() сразу после dydt(). Он пишет
// в те же самые d_mem->conc/fwd_rates/rev_rates/pres_mod/spec_rates, что и dydt
// (см. jacob.cu) - то есть полностью перезаписывал их своими значениями ПОСЛЕ
// dydt(), и в pyjac_output.txt утекали именно они, а не значения dydt().
// eval_jacob для GRIMech сгенерирован ТОЛЬКО в CONP-варианте (в jacob.cu нет ни
// одного #ifdef CONV/CONP - grep по всему файлу пуст), то есть он всегда трактует
// свой 2-й аргумент как ДАВЛЕНИЕ через eval_conc(), независимо от того, что
// header.cuh для этого механизма определяет CONV. Мы передаём сюда rho (~0.26
// кг/м3) - eval_jacob использует её как pres (~0.26 Па, почти вакуум) вместо
// 101325 Па, из-за чего conc[] получались в ~101325/0.26 ~ 390000 раз меньше
// правильных. Для чистого сравнения dydt() с Cantera eval_jacob здесь не нужен -
// он не убирает данные для diff'а, а лишь дописывает результат (см. чуть ниже).
__global__ void probe_kernel(mechanism_memory* d_mem, double t, double param)
{
    dydt(t, param, d_mem->y, d_mem->dy, d_mem);
}

static bool readState(const std::string& path, double& T0, double& P0, std::vector<double>& Xi)
{
    std::ifstream in(path);
    if (!in) return false;

    std::vector<std::string> dataLines;
    std::string line;
    while (std::getline(in, line) && dataLines.size() < 2)
    {
        size_t hash = line.find('#');
        if (hash != std::string::npos) line = line.substr(0, hash);
        std::istringstream check(line);
        std::string tok;
        bool hasData = false;
        while (check >> tok) hasData = true;
        if (hasData) dataLines.push_back(line);
    }
    if (dataLines.size() < 2) return false;

    {
        std::istringstream iss(dataLines[0]);
        if (!(iss >> T0 >> P0)) return false;
    }
    Xi.assign(NSP, 0.0);
    {
        std::istringstream iss(dataLines[1]);
        for (int i = 0; i < NSP; ++i)
            if (!(iss >> Xi[i])) return false;
    }
    return true;
}

int main(int argc, char** argv)
{
    std::string path = argc > 1 ? argv[1] : "state.txt";

    double T0, P0;
    std::vector<double> Xi;
    if (!readState(path, T0, P0, Xi))
    {
        fprintf(stderr, "Не удалось прочитать состояние из %s\n", path.c_str());
        return 1;
    }

    double Xsum = 0.0;
    for (int i = 0; i < NSP; ++i) Xsum += Xi[i];
    for (int i = 0; i < NSP; ++i) Xi[i] /= Xsum;

    double Yi[NSP - 1];
    mole2mass(Xi.data(), Yi);

    double rho = getDensity(T0, P0, Xi.data());

    double y_N = 1.0;
    for (int i = 0; i < NSP - 1; ++i) y_N -= Yi[i];

    printf("=== input state ===\n");
    printf("T0  = %.10f K\n", T0);
    printf("P0  = %.10f Pa\n", P0);
    printf("rho (getDensity, предполагается kg/m3) = %.15e\n\n", rho);

    printf("mole fractions (ненулевые):\n");
    for (int i = 0; i < NSP; ++i)
        if (Xi[i] != 0.0) printf("  X[%s] = %.10f\n", SPECIES_NAMES[i], Xi[i]);

    printf("\nmass fractions (ненулевые, %s - неявный/последний вид):\n", SPECIES_NAMES[NSP - 1]);
    for (int i = 0; i < NSP - 1; ++i)
        if (Yi[i] != 0.0) printf("  Y[%s] = %.10f\n", SPECIES_NAMES[i], Yi[i]);
    printf("  Y[%s] (= 1 - sum остальных) = %.10f\n\n", SPECIES_NAMES[NSP - 1], y_N);

    // ---- один "система" на GPU ----
    mechanism_memory* h_mem = (mechanism_memory*)malloc(sizeof(mechanism_memory));
    mechanism_memory* d_mem = nullptr;
    initialize_gpu_memory(1, &h_mem, &d_mem);

    double y_host[NSP];
    y_host[0] = T0;
    for (int i = 0; i < NSP - 1; ++i) y_host[i + 1] = Yi[i];

    CUDA_CHECK(cudaMemcpy(h_mem->y, y_host, NSP * sizeof(double), cudaMemcpyHostToDevice));

    // ВАЖНО: это ровно то же значение, которое kodes::GRIMESHSystem::derivatives
    // сейчас передаёт вторым аргументом в dydt()/eval_jacob() (после вашего фикса -
    // res->parameters(workIndex), т.е. по построению совпадает с rho из
    // set_same_initial_conditions). header.cuh для этого механизма определяет
    // CONV, так что второй аргумент - плотность (кг/м3), не давление.
    double param = rho;

    probe_kernel<<<1, 1, 4 * 1 * sizeof(double)>>>(d_mem, 0.0, param);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    static double conc[NSP];
    static double fwd[FWD_RATES];
    static double rev[REV_RATES];
    static double pmod[PRES_MOD_RATES];
    static double spec[NSP];
    double dy[NN];

    CUDA_CHECK(cudaMemcpy(conc, h_mem->conc, NSP * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(fwd, h_mem->fwd_rates, FWD_RATES * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(rev, h_mem->rev_rates, REV_RATES * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(pmod, h_mem->pres_mod, PRES_MOD_RATES * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(spec, h_mem->spec_rates, NSP * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dy, h_mem->dy, NSP * sizeof(double), cudaMemcpyDeviceToHost));

    printf("=== pyJac: промежуточные величины (param = %.15e) ===\n\n", param);

    printf("conc[i] - должно быть в kmol/m3, ЕСЛИ вся цепочка (getDensity/eval_conc_rho/\n");
    printf("get_rxn_pres_mod, все использующие R=8314.4621 J/(kmol*K)) самосогласована:\n");
    for (int i = 0; i < NSP; ++i)
        printf("  conc[%s] = %.15e\n", SPECIES_NAMES[i], conc[i]);

    printf("\nspec_rates[i] - чистая скорость образования вида (моль/объём/время);\n");
    printf("сравнивать напрямую с Cantera gas.net_production_rates():\n");
    for (int i = 0; i < NSP - 1; ++i)
        printf("  spec_rates[%s] = %.15e\n", SPECIES_NAMES[i], spec[i]);
    printf("  spec_rates[%s] (последний вид, dy_N) = %.15e\n", SPECIES_NAMES[NSP - 1], spec[NSP - 1]);

    printf("\ndy/dt - итоговые производные, ровно то, что видит интегратор:\n");
    printf("  dT/dt = %.15e\n", dy[0]);
    for (int i = 0; i < NSP - 1; ++i)
        printf("  dY[%s]/dt = %.15e\n", SPECIES_NAMES[i], dy[i + 1]);

    printf("\n--- сырые fwd_rates/rev_rates (все %d/%d), для точечной сверки ---\n", FWD_RATES, REV_RATES);
    printf("--- ВНИМАНИЕ: индекс реакции здесь совпадает с индексом Cantera gas.reaction(i)\n");
    printf("--- только если pyJac не менял порядок реакций из исходного механизма ---\n");
    for (int i = 0; i < FWD_RATES; ++i)
        printf("  fwd_rates[%d] = %.15e\n", i, fwd[i]);
    for (int i = 0; i < REV_RATES; ++i)
        printf("  rev_rates[%d] = %.15e\n", i, rev[i]);
    for (int i = 0; i < PRES_MOD_RATES; ++i)
        printf("  pres_mod[%d] = %.15e\n", i, pmod[i]);

    free_gpu_memory(&h_mem, &d_mem);
    free(h_mem);
    return 0;
}
