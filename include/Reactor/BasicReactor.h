#pragma once

namespace kodes {

template<typename T, size_t N_Species>
struct BasicReactor {
    T concentrations[N_Species];   // Текущие концентрации
    T temperature;                 // Температура (может меняться, если не изотермический процесс)
    T density;                     // Плотность
    T source_terms[N_Species];     // Вычисленные скорости реакций (output)
    
    // Параметры конкретной ячейки (могут отличаться от ячейки к ячейке)
    T pressure;                    // Давление (если нужно)
    
    // Можно добавить флаги активности
    bool is_active;
};

// Вспомогательный класс для управления массивом реакторов на GPU
// Это не полиморфный класс, а контейнер
template<typename T, size_t N_Species>
class ReactorBatch {
private:
    T* d_concentrations;      // [N_Reactors * N_Species] на GPU
    T* d_temperature;         // [N_Reactors]
    T* d_source_terms;        // [N_Reactors * N_Species]
    // ... остальные массивы SoA
    
    size_t num_reactors;
    size_t capacity;
    
public:
    // Управление памятью на GPU
    void resize(size_t new_capacity);
    void upload(size_t reactor_idx, const BasicReactor<T, N_Species>& host_reactor);
    void download(size_t reactor_idx, BasicReactor<T, N_Species>& host_reactor);
    
    // Доступ к сырым указателям для ядер CUDA
    T* get_concentrations() { return d_concentrations; }
    T* get_temperatures() { return d_temperature; }
    // ...
};

} // namespace kinetics