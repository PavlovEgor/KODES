#pragma once
#include "BasicReactor.h"

namespace kodes {
namespace solvers {

// Базовый интерфейс (абстрактный) для полиморфизма на уровне системы
template<typename T, size_t N_Species>
class ISolver {
public:
    virtual ~ISolver() = default;
    
    // Решает систему для батча реакторов
    // dt - шаг по времени (может быть свой для каждой ячейки или общий)
    virtual void solve(
        ReactorBatch<T, N_Species>& batch,
        T dt,
        cudaStream_t stream = 0
    ) = 0;
    
    // Возвращает оценку следующего шага для адаптивного шага (опционально)
    virtual T estimate_error(const ReactorBatch<T, N_Species>& batch) = 0;
};

// Конкретная реализация: явный метод Рунге-Кутты 4-го порядка
template<typename T, size_t N_Species>
class RK4Solver : public ISolver<T, N_Species> {
private:
    // Временные буферы на GPU для стадий RK4
    T* d_k1;
    T* d_k2;
    T* d_k3;
    T* d_k4;
    size_t buffer_size;
    
public:
    RK4Solver(size_t max_reactors);
    ~RK4Solver();
    
    void solve(ReactorBatch<T, N_Species>& batch, T dt, cudaStream_t stream) override;
    T estimate_error(const ReactorBatch<T, N_Species>& batch) override { return 0; /* Not implemented for RK4 */ }
};

// Неявный метод: Кранка-Николсона / BDF (потребует решения линейных систем)
template<typename T, size_t N_Species>
class CVODESolver : public ISolver<T, N_Species> {
    // Будет содержать указатели на функции вычисления якобиана (тоже на GPU)
    // И, вероятно, использовать cuSOLVER или собственные решатели (bicgstab, etc.)
public:
    CVODESolver(size_t max_reactors);
    ~CVODESolver();
    
    void solve(ReactorBatch<T, N_Species>& batch, T dt, cudaStream_t stream) override;
    T estimate_error(const ReactorBatch<T, N_Species>& batch) override;
};

} // namespace solvers
} // namespace kinetics