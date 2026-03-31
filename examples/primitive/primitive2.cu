#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <chrono>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)


// Structure representing the chemical state of a single reactor cell
typedef struct {
    int numberOfComponents;     // Number of chemical components in this cell
    double* concentrations;     // Array of concentrations for each component
} BasicChemistryState;

// Structure containing the right-hand side function parameters
// For the ODE system: dx/dt = f(x) = -k * x
typedef struct {
    double k;                   // Reaction rate constant
} ChemistryRHS;

// Euler method solver parameters
typedef struct {
    double dt;      // Desired time step
    double tEnd;    // Final simulation time
} EulerSolver;

/**
 * CUDA kernel that solves a system of ODEs using the Euler method
 * Each thread processes one reactor cell independently
 * 
 * @param solver      Euler solver parameters (time step, final time)
 * @param rhs         Right-hand side function parameters
 * @param cells       Array of reactor cell states
 * @param numberOfCells Total number of cells to process
 */
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
        
        // Integrate until reaching the final time
        while (currentTime < solver.tEnd - 1e-12) {
            // Determine the time step for current iteration
            double dt = solver.dt;
            double remainingTime = solver.tEnd - currentTime;
            if (dt > remainingTime) {
                dt = remainingTime;  // Adjust last step to exactly reach tEnd
            }
            
            // Compute derivatives and update concentrations using Euler method
            double k = rhs.k;
            for (int i = 0; i < nComp; i++) {
                // Euler method: x_new = x_old + dt * (-k * x_old)
                cell->concentrations[i] += dt * (-k * cell->concentrations[i]);
            }
            
            currentTime += dt;
        }
    }
}

int main() {
    // Problem size configuration
    int numberOfCells = 1 << 20;        // ~1 million cells
    int numberOfComponents = 50;         // Chemical components per cell
    
    // Timing variables
    double timeH2D = 0.0;               // Host to Device copy time
    double timeKernel = 0.0;            // Kernel execution time
    double timeD2H = 0.0;               // Device to Host copy time
    double totalTime = 0.0;             // Total time
    
    EulerSolver solver;
    ChemistryRHS rhs;
    
    // Solver configuration
    solver.dt = 0.01;      // Time step
    solver.tEnd = 1.0;     // Final simulation time
    rhs.k = 1.0;           // Reaction rate constant
    
    // Analytical solution: x(t) = x(0) * exp(-k*t)
    double expectedFactor = std::exp(-rhs.k * solver.tEnd);
    std::cout << "=== Configuration ===" << std::endl;
    std::cout << "Number of cells: " << numberOfCells << std::endl;
    std::cout << "Components per cell: " << numberOfComponents << std::endl;
    std::cout << "Total elements: " << (long long)numberOfCells * numberOfComponents << std::endl;
    std::cout << "Memory per cell: " << (numberOfComponents * sizeof(double)) / 1024.0 << " KB" << std::endl;
    std::cout << "Total memory: " << (numberOfCells * numberOfComponents * sizeof(double)) / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "Time step: " << solver.dt << std::endl;
    std::cout << "Final time: " << solver.tEnd << std::endl;
    std::cout << "Number of steps: " << (int)(solver.tEnd / solver.dt) << std::endl;
    std::cout << "Expected decay factor: " << expectedFactor << std::endl;
    std::cout << std::endl;
    
    // Start total timing
    auto total_start = std::chrono::high_resolution_clock::now();
    
    // ========================================================================
    // Host memory allocation and initialization
    // ========================================================================
    BasicChemistryState* h_cellsPull = (BasicChemistryState*)malloc(numberOfCells * sizeof(BasicChemistryState));
    BasicChemistryState* h_cellsPull_res = (BasicChemistryState*)malloc(numberOfCells * sizeof(BasicChemistryState));
    BasicChemistryState* d_cellsPull;
    
    if (!h_cellsPull || !h_cellsPull_res) {
        std::cerr << "Failed to allocate host memory" << std::endl;
        return 1;
    }
    
    // Initialize initial conditions on host
    std::cout << "Initializing host data..." << std::endl;
    for (int i = 0; i < numberOfCells; i++) {
        h_cellsPull[i].numberOfComponents = numberOfComponents;
        h_cellsPull[i].concentrations = (double*)malloc(numberOfComponents * sizeof(double));
        
        if (!h_cellsPull[i].concentrations) {
            std::cerr << "Failed to allocate concentrations for cell " << i << std::endl;
            return 1;
        }
        
        // Set initial concentrations: different for each cell, uniform across components
        for (int j = 0; j < numberOfComponents; j++) {
            h_cellsPull[i].concentrations[j] = (double)(i + 1);
        }
    }
    
    // ========================================================================
    // Device memory allocation
    // ========================================================================
    std::cout << "Allocating device memory..." << std::endl;
    
    // Allocate device memory for cell structures array
    CUDA_CHECK(cudaMalloc(&d_cellsPull, numberOfCells * sizeof(BasicChemistryState)));
    
    // Allocate device memory for each concentration array
    double** d_concentrations = (double**)malloc(numberOfCells * sizeof(double*));
    
    // Start timing Host->Device transfer
    auto h2d_start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < numberOfCells; i++) {
        CUDA_CHECK(cudaMalloc(&d_concentrations[i], numberOfComponents * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_concentrations[i], h_cellsPull[i].concentrations, 
                              numberOfComponents * sizeof(double), cudaMemcpyHostToDevice));
    }
    
    auto h2d_end = std::chrono::high_resolution_clock::now();
    timeH2D = std::chrono::duration<double>(h2d_end - h2d_start).count();
    
    // Create host-side structures with correct device pointers for copying to GPU
    BasicChemistryState* h_cellsPull_devicePtrs = (BasicChemistryState*)malloc(numberOfCells * sizeof(BasicChemistryState));
    for (int i = 0; i < numberOfCells; i++) {
        h_cellsPull_devicePtrs[i].numberOfComponents = numberOfComponents;
        h_cellsPull_devicePtrs[i].concentrations = d_concentrations[i];
    }
    
    // Copy structures to device
    CUDA_CHECK(cudaMemcpy(d_cellsPull, h_cellsPull_devicePtrs, numberOfCells * sizeof(BasicChemistryState), 
                          cudaMemcpyHostToDevice));
    
    // ========================================================================
    // Kernel execution
    // ========================================================================
    // Configure grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (numberOfCells + threadsPerBlock - 1) / threadsPerBlock;
    
    std::cout << "Launching kernel with " << blocksPerGrid << " blocks, " 
              << threadsPerBlock << " threads per block" << std::endl;
    
    // Start timing kernel execution
    auto kernel_start = std::chrono::high_resolution_clock::now();
    
    // Launch kernel
    SolveReactorsSystemEuler<<<blocksPerGrid, threadsPerBlock>>>(solver, rhs, d_cellsPull, numberOfCells);
    CUDA_CHECK(cudaDeviceSynchronize());  
    CUDA_CHECK(cudaGetLastError());
    
    auto kernel_end = std::chrono::high_resolution_clock::now();
    timeKernel = std::chrono::duration<double>(kernel_end - kernel_start).count();
    
    // ========================================================================
    // Copy results back to host
    // ========================================================================
    // Start timing Device->Host transfer
    auto d2h_start = std::chrono::high_resolution_clock::now();
    
    // First copy the structures (concentrations pointers remain device pointers)
    CUDA_CHECK(cudaMemcpy(h_cellsPull_res, d_cellsPull, numberOfCells * sizeof(BasicChemistryState), 
                          cudaMemcpyDeviceToHost));
    
    // Then copy the actual concentration data
    for (int i = 0; i < numberOfCells; i++) {
        h_cellsPull_res[i].concentrations = (double*)malloc(numberOfComponents * sizeof(double));
        h_cellsPull_res[i].numberOfComponents = numberOfComponents;
        
        CUDA_CHECK(cudaMemcpy(h_cellsPull_res[i].concentrations, d_concentrations[i],
                              numberOfComponents * sizeof(double), cudaMemcpyDeviceToHost));
    }
    
    auto d2h_end = std::chrono::high_resolution_clock::now();
    timeD2H = std::chrono::duration<double>(d2h_end - d2h_start).count();
    
    auto total_end = std::chrono::high_resolution_clock::now();
    totalTime = std::chrono::duration<double>(total_end - total_start).count();
    
    // ========================================================================
    // Verification and error analysis
    // ========================================================================
    std::cout << "\nVerifying results..." << std::endl;
    
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
    
    // ========================================================================
    // Performance and results summary
    // ========================================================================
    std::cout << "\n=== Performance Summary ===" << std::endl;
    std::cout << "Host to Device transfer time: " << timeH2D * 1000 << " ms" << std::endl;
    std::cout << "Kernel execution time: " << timeKernel * 1000 << " ms" << std::endl;
    std::cout << "Device to Host transfer time: " << timeD2H * 1000 << " ms" << std::endl;
    std::cout << "Total time: " << totalTime * 1000 << " ms" << std::endl;
    std::cout << std::endl;
    
    // Calculate performance metrics
    long long totalOps = (long long)numberOfCells * numberOfComponents * (int)(solver.tEnd / solver.dt);
    double gflops = (totalOps * 2.0) / (timeKernel * 1e9); // 2 operations per update (mul + add)
    double bandwidth = (numberOfCells * numberOfComponents * sizeof(double) * 2) / (timeKernel * 1e9); // Read + Write
    
    std::cout << "=== Performance Metrics ===" << std::endl;
    std::cout << "Total floating-point operations: " << totalOps << " (approx)" << std::endl;
    std::cout << "Kernel performance: " << gflops << " GFLOPS" << std::endl;
    std::cout << "Memory bandwidth: " << bandwidth << " GB/s" << std::endl;
    std::cout << std::endl;
    
    std::cout << "=== Accuracy Results ===" << std::endl;
    std::cout << "Max absolute error: " << maxError << std::endl;
    std::cout << "Max relative error: " << maxRelativeError * 100 << " %" << std::endl;
    std::cout << std::endl;
    
    // Sample output of first few cells
    std::cout << "=== Sample Results (First 5 cells) ===" << std::endl;
    for (int i = 0; i < std::min(5, numberOfCells); i++) {
        double initialValue = (double)(i + 1);
        double expected = initialValue * expectedFactor;
        std::cout << "Cell " << i << ": initial=" << initialValue 
                  << ", computed=" << h_cellsPull_res[i].concentrations[0]
                  << ", expected=" << expected
                  << ", error=" << fabs(h_cellsPull_res[i].concentrations[0] - expected) << std::endl;
    }
    
    // ========================================================================
    // Cleanup
    // ========================================================================
    std::cout << "\nCleaning up memory..." << std::endl;
    
    for (int i = 0; i < numberOfCells; i++) {
        free(h_cellsPull[i].concentrations);      // Free host initial condition memory
        free(h_cellsPull_res[i].concentrations);  // Free host result memory
        CUDA_CHECK(cudaFree(d_concentrations[i])); // Free device memory
    }
    free(h_cellsPull);
    free(h_cellsPull_res);
    free(d_concentrations);
    free(h_cellsPull_devicePtrs);                 // Free temporary pointer array
    CUDA_CHECK(cudaFree(d_cellsPull));
    
    std::cout << "Done!" << std::endl;
    
    return 0;
}