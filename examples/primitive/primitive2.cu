#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)


typedef struct {
    int numberOfComponents;
    double* concentrations;
} BasicChemistryState;

typedef struct {
    double k;
} ChemistryRHS;

typedef struct {
    double dt;      // желаемый шаг по времени
    double tEnd;    // конечное время
} EulerSolver;

__global__ 
void SolveReactorsSystemEuler(
    EulerSolver solver, 
    const ChemistryRHS rhs,
    BasicChemistryState* cells, 
    int numberOfCells) {
    
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < numberOfCells) {
        BasicChemistryState* cell = &cells[idx];
        double currentTime = 0.0;
        int nComp = cell->numberOfComponents;
        
        // Выполняем интегрирование пока не достигнем tEnd
        while (currentTime < solver.tEnd - 1e-12) {
            // Определяем шаг для текущей итерации
            double dt = solver.dt;
            double remainingTime = solver.tEnd - currentTime;
            if (dt > remainingTime) {
                dt = remainingTime;
            }
            
            // Вычисляем производные и обновляем концентрации
            double k = rhs.k;
            for (int i = 0; i < nComp; i++) {
                // Метод Эйлера: x_new = x_old + dt * (-k * x_old)
                cell->concentrations[i] += dt * (-k * cell->concentrations[i]);
            }
            
            currentTime += dt;
        }
    }
}

int main() {
    int numberOfCells = 1 << 20;
    int numberOfComponents = 50;

    EulerSolver solver;
    ChemistryRHS rhs;

    // Настройка решателя
    solver.dt = 0.01;      // желаемый шаг
    solver.tEnd = 1.0;    // конечное время
    rhs.k = 1.0;          // константа скорости
    
    // Аналитическое решение: x(t) = x(0) * exp(-k*t)
    double expectedFactor = std::exp(-rhs.k * solver.tEnd);
    std::cout << "Expected factor: " << expectedFactor << std::endl;

    BasicChemistryState* h_cellsPull = (BasicChemistryState*)malloc(numberOfCells * sizeof(BasicChemistryState));
    BasicChemistryState* h_cellsPull_res = (BasicChemistryState*)malloc(numberOfCells * sizeof(BasicChemistryState));
    BasicChemistryState* d_cellsPull;
    
    if (!h_cellsPull || !h_cellsPull_res) {
        std::cerr << "Failed to allocate host memory" << std::endl;
        return 1;
    }
    
    // Инициализация начальных условий на хосте
    for (int i = 0; i < numberOfCells; i++) {
        h_cellsPull[i].numberOfComponents = numberOfComponents;
        h_cellsPull[i].concentrations = (double*)malloc(numberOfComponents * sizeof(double));
        
        if (!h_cellsPull[i].concentrations) {
            std::cerr << "Failed to allocate concentrations for cell " << i << std::endl;
            return 1;
        }
        
        for (int j = 0; j < numberOfComponents; j++) {
            h_cellsPull[i].concentrations[j] = (double)(i + 1); // начальные значения от 1 до numberOfCells
        }
    }

    // Выделение памяти на устройстве для массива структур
    CUDA_CHECK(cudaMalloc(&d_cellsPull, numberOfCells * sizeof(BasicChemistryState)));
    
    // Выделяем память для каждого массива концентраций на устройстве
    double** d_concentrations = (double**)malloc(numberOfCells * sizeof(double*));
    
    for (int i = 0; i < numberOfCells; i++) {
        CUDA_CHECK(cudaMalloc(&d_concentrations[i], numberOfComponents * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_concentrations[i], h_cellsPull[i].concentrations, 
                              numberOfComponents * sizeof(double), cudaMemcpyHostToDevice));
    }
    
    // Создаем копию структур на хосте с правильными указателями на device-память
    BasicChemistryState* h_cellsPull_devicePtrs = (BasicChemistryState*)malloc(numberOfCells * sizeof(BasicChemistryState));
    for (int i = 0; i < numberOfCells; i++) {
        h_cellsPull_devicePtrs[i].numberOfComponents = numberOfComponents;
        h_cellsPull_devicePtrs[i].concentrations = d_concentrations[i];
    }
    
    // Копируем структуры на устройство
    CUDA_CHECK(cudaMemcpy(d_cellsPull, h_cellsPull_devicePtrs, numberOfCells * sizeof(BasicChemistryState), 
                          cudaMemcpyHostToDevice));

    // Настройка сетки
    int threadsPerBlock = 256;
    int blocksPerGrid = (numberOfCells + threadsPerBlock - 1) / threadsPerBlock;
    
    std::cout << "Launching kernel with " << blocksPerGrid << " blocks, " 
              << threadsPerBlock << " threads per block" << std::endl;
    
    // Запуск ядра
    SolveReactorsSystemEuler<<<blocksPerGrid, threadsPerBlock>>>(solver, rhs, d_cellsPull, numberOfCells);
    CUDA_CHECK(cudaDeviceSynchronize());  
    CUDA_CHECK(cudaGetLastError());       

    // Копируем результаты обратно на хост
    // Сначала копируем структуры (но указатели concentrations останутся device-указателями)
    CUDA_CHECK(cudaMemcpy(h_cellsPull_res, d_cellsPull, numberOfCells * sizeof(BasicChemistryState), 
                          cudaMemcpyDeviceToHost));
    
    // Теперь копируем сами данные концентраций
    for (int i = 0; i < numberOfCells; i++) {
        h_cellsPull_res[i].concentrations = (double*)malloc(numberOfComponents * sizeof(double));
        h_cellsPull_res[i].numberOfComponents = numberOfComponents;
        
        CUDA_CHECK(cudaMemcpy(h_cellsPull_res[i].concentrations, d_concentrations[i],
                              numberOfComponents * sizeof(double), cudaMemcpyDeviceToHost));
    }

    // Проверка результатов
    double maxError = 0.0;
    double maxRelativeError = 0.0;
    
    for (int i = 0; i < numberOfCells; i++) {
        double initialValue = (double)(i + 1);
        double expected = initialValue * expectedFactor;
        
        for (int j = 0; j < numberOfComponents; j++) {
            double actual = h_cellsPull_res[i].concentrations[j];
            double absoluteError = fabs(actual - expected);
            double relativeError = absoluteError / fabs(expected);
            
            maxError = fmax(maxError, absoluteError);
            maxRelativeError = fmax(maxRelativeError, relativeError);
        }
    }

    std::cout << "\n=== Results ===" << std::endl;
    std::cout << "Number of cells: " << numberOfCells << std::endl;
    std::cout << "Number of components per cell: " << numberOfComponents << std::endl;
    std::cout << "dt (requested): " << solver.dt << std::endl;
    std::cout << "tEnd: " << solver.tEnd << std::endl;
    std::cout << "Number of steps: " << (int)(solver.tEnd / solver.dt) << std::endl;
    std::cout << "Expected factor: " << expectedFactor << std::endl;
    std::cout << "Max absolute error: " << maxError << std::endl;
    std::cout << "Max relative error: " << maxRelativeError << std::endl;
    
    // Вывод первых нескольких результатов для проверки
    std::cout << "\nFirst 5 cells results:" << std::endl;
    for (int i = 0; i < std::min(5, numberOfCells); i++) {
        double initialValue = (double)(i + 1);
        double expected = initialValue * expectedFactor;
        std::cout << "Cell " << i << ": initial=" << initialValue 
                  << ", computed=" << h_cellsPull_res[i].concentrations[0]
                  << ", expected=" << expected << std::endl;
    }

    // Очистка памяти
    for (int i = 0; i < numberOfCells; i++) {
        free(h_cellsPull[i].concentrations);  // Освобождаем host-память начальных условий
        free(h_cellsPull_res[i].concentrations); // Освобождаем host-память результатов
        CUDA_CHECK(cudaFree(d_concentrations[i])); // Освобождаем device-память
    }
    free(h_cellsPull);
    free(h_cellsPull_res);
    free(d_concentrations);
    free(h_cellsPull_devicePtrs); // Освобождаем временный массив указателей
    CUDA_CHECK(cudaFree(d_cellsPull));

    return 0;
}