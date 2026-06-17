
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda/cmath>
#include <cuda_runtime.h>
#include "euler.cuh"
#include <iostream>
#include <chrono> 


int main(){

    auto start_total = std::chrono::high_resolution_clock::now();

    scalar xEnd = 321.8122;
    scalar xStart = 0.0;

    ODEVectors vectors;
    vectors.sizeOfSystem = 8;
    vectors.numOfSystems = 1 << 13;
    size_t sizeOfData = vectors.sizeOfSystem * vectors.numOfSystems * sizeof(scalar);

    auto start_alloc = std::chrono::high_resolution_clock::now();
    cudaMallocManaged(&vectors.data, sizeOfData);
    auto end_alloc = std::chrono::high_resolution_clock::now();

    auto duration_alloc = std::chrono::duration_cast<std::chrono::microseconds>(end_alloc - start_alloc);
    std::cout << "Время выделения памяти: " << duration_alloc.count() << " мкс" << std::endl;

    auto start_init = std::chrono::high_resolution_clock::now();
    init(&vectors);
    auto end_init = std::chrono::high_resolution_clock::now();
    auto duration_init = std::chrono::duration_cast<std::chrono::microseconds>(end_init - start_init);
    std::cout << "Время инициализации: " << duration_init.count() << " мкс" << std::endl;

    std::cout << std::endl;
    for (int j=0; j < vectors.sizeOfSystem; j++)
    {
        std::cout << vectors.data[0 * vectors.sizeOfSystem + j] << " ";
    } std::cout << std::endl; std::cout << std::endl;

    int threads = 256;
    int blocks = cuda::ceil_div(vectors.numOfSystems, threads);
    
    cudaEvent_t start_kernel, stop_kernel;
    cudaEventCreate(&start_kernel);
    cudaEventCreate(&stop_kernel);
    
    cudaEventRecord(start_kernel);
    euler_solve<<<blocks, threads>>>(vectors.data, vectors.numOfSystems, xStart, xEnd);
    cudaEventRecord(stop_kernel);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }
    
    
    cudaEventSynchronize(stop_kernel);
    float kernel_time_ms = 0;
    cudaEventElapsedTime(&kernel_time_ms, start_kernel, stop_kernel);
    std::cout << "Время выполнения ядра: " << kernel_time_ms << " мс" << std::endl;
    
    cudaEventDestroy(start_kernel);
    cudaEventDestroy(stop_kernel);

    for (int i=vectors.numOfSystems-1; i < vectors.numOfSystems; ++i){
    for (int j=0; j < vectors.sizeOfSystem; j++)
    {
        std::cout << vectors.data[i * vectors.sizeOfSystem + j] << " ";
    } std::cout << std::endl;
    } std::cout << std::endl;

    std::cout << "0.000737131 0.000144249 5.88873e-05 0.00117565 0.00238636 0.00623897 0.00285 0.00285" <<std::endl;
    
    
    auto start_free = std::chrono::high_resolution_clock::now();
    cudaFree(vectors.data);
    auto end_free = std::chrono::high_resolution_clock::now();
    auto duration_free = std::chrono::duration_cast<std::chrono::microseconds>(end_free - start_free);
    std::cout << "Время освобождения памяти: " << duration_free.count() << " мкс" << std::endl;

    // Общее время выполнения
    auto end_total = std::chrono::high_resolution_clock::now();
    auto duration_total = std::chrono::duration_cast<std::chrono::milliseconds>(end_total - start_total);
    std::cout << "Общее время выполнения: " << duration_total.count() << " мс" << std::endl;

    return 0;
}

void init(ODEVectors* vectors)
{
    for (int i=0; i < vectors -> numOfSystems; ++i)
    {
        for (int j=0; j<vectors -> sizeOfSystem; ++j)
        {
            vectors -> data[i * vectors -> sizeOfSystem + j] = 0;
        }
        vectors -> data[i * vectors -> sizeOfSystem + 0] = 1.0;
        vectors -> data[i * vectors -> sizeOfSystem + 7] = 0.0057;
    }
}

__device__
void derivatives(const scalar x, const scalar* y, scalar* dydx)
{
    scalar y1 = y[0];
    scalar y2 = y[1];
    scalar y3 = y[2];
    scalar y4 = y[3];
    scalar y5 = y[4];
    scalar y6 = y[5];
    scalar y7 = y[6];
    scalar y8 = y[7];
    
    dydx[0] = -1.71 * y1 + 0.43 * y2 + 8.32 * y3 + 0.0007;
    dydx[1] = 1.71 * y1 - 8.75 * y2;
    dydx[2] = -10.03 * y3 + 0.43 * y4 + 0.035 * y5;
    dydx[3] = 8.32 * y2 + 1.71 * y3 - 1.12 * y4;
    dydx[4] = -1.745 * y5 + 0.43 * y6 + 0.43 * y7;
    dydx[5] = -280.0 * y6 * y8 + 0.69 * y4 + 1.71 * y5 - 0.43 * y6 + 0.69 * y7;
    dydx[6] = 280.0 * y6 * y8 - 1.81 * y7;
    dydx[7] = -280 * y6 * y8 + 1.81 * y7;
}

__device__
void jacobian(const scalar x, const scalar* y, scalar* dfdx, scalar* dfdy)
{
    size_t sizeOfSystem = 8;
    
    // df/dx = 0 for autonomous system
    for (size_t i = 0; i < sizeOfSystem; ++i)
    {
        dfdx[i] = 0.0;
    }
    
    // Initialize Jacobian matrix with zeros
    for (size_t i = 0; i < sizeOfSystem; ++i)
    {
        for (size_t j = 0; j < sizeOfSystem; ++j)
        {
            dfdy[i*sizeOfSystem + j] = 0.0;
        }
    }
    
    scalar y6 = y[5];
    scalar y8 = y[7];
    
    // Row 0: derivatives of y1'
    dfdy[0*sizeOfSystem + 0] = -1.71;
    dfdy[0*sizeOfSystem + 1] = 0.43;
    dfdy[0*sizeOfSystem + 2] = 8.32;

    // Row 1: derivatives of y2'
    dfdy[1*sizeOfSystem + 0] = 1.71;
    dfdy[1*sizeOfSystem + 1] = -8.75;

    // Row 2: derivatives of y3'
    dfdy[2*sizeOfSystem + 2] = -10.03;
    dfdy[2*sizeOfSystem + 3] = 0.43;
    dfdy[2*sizeOfSystem + 4] = 0.035;

    // Row 3: derivatives of y4'
    dfdy[3*sizeOfSystem + 1] = 8.32;
    dfdy[3*sizeOfSystem + 2] = 1.71;
    dfdy[3*sizeOfSystem + 3] = -1.12;

    // Row 4: derivatives of y5'
    dfdy[4*sizeOfSystem + 4] = -1.745;
    dfdy[4*sizeOfSystem + 5] = 0.43;
    dfdy[4*sizeOfSystem + 6] = 0.43;

    // Row 5: derivatives of y6'
    dfdy[5*sizeOfSystem + 3] = 0.69;
    dfdy[5*sizeOfSystem + 4] = 1.71;
    dfdy[5*sizeOfSystem + 5] = -280.0 * y8 - 0.43;
    dfdy[5*sizeOfSystem + 6] = 0.69;
    dfdy[5*sizeOfSystem + 7] = -280.0 * y6;

    // Row 6: derivatives of y7'
    dfdy[6*sizeOfSystem + 5] = 280.0 * y8;
    dfdy[6*sizeOfSystem + 6] = -1.81;
    dfdy[6*sizeOfSystem + 7] = 280.0 * y6;

    // Row 7: derivatives of y8'
    dfdy[7*sizeOfSystem + 5] = -280 * y8;
    dfdy[7*sizeOfSystem + 6] = 1.81;
    dfdy[7*sizeOfSystem + 7] = -280 * y6;
}

__device__
scalar solve(const scalar x0, const scalar* y0, const scalar* dydx0, scalar dx, scalar* y)
{
    scalar err_[8];

    for(size_t i=0; i < sizeOfSystem; ++i)
    {
        err_[i] = dx*dydx0[i];
        y[i] = y0[i] + err_[i];
    }

    return normalizeError(y0, y, err_, &sizeOfSystem, &absTol_, &relTol_);
}

__global__
void euler_solve(scalar* data, const label numOfSystems, const scalar xStart, const scalar xEnd)
{
    int workIndex = threadIdx.x + blockIdx.x*blockDim.x;

    if (workIndex < numOfSystems)
    {
        scalar dydx0_[8];
        scalar yTemp_[8];
        scalar* y = (scalar*)(data + workIndex * 8);

        scalar dx = xEnd;
        scalar err = 0.0;
        scalar x = xStart;

    do{
        derivatives(x, y, dydx0_);

        do
        {
            err = solve(x, y, dydx0_, dx, yTemp_);

            if (err > 1)
            {
                scalar scale = max(safeScale_*pow(err, -alphaDec_), minScale_);
                dx *= scale;
            }

        } while (err > 1);

        x += dx;

        for (int i=0; i < sizeOfSystem; ++i)
        {
            y[i] = yTemp_[i];
        }

        if (err > pow(maxScale_/safeScale_, -1.0/alphaInc_))
        {
            scalar scale = safeScale_*pow(err, -alphaInc_);
            dx = clamp(scale, minScale_, maxScale_)*dx;
        }
        else
        {
            dx = safeScale_*maxScale_*dx;
        }

    } while (x < xEnd);

    }
}


