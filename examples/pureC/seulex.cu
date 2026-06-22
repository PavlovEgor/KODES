
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda/cmath>
#include <cuda_runtime.h>
#include "seulex.cuh"
#include <iostream>
#include <chrono> 


int main(){

    auto start_total = std::chrono::high_resolution_clock::now();

    scalar xEnd = 321.8122;
    stepState step(xEnd);

    ODEVectors vectors;
    scalar* dev_data;
    vectors.sizeOfSystem = 8;
    vectors.numOfSystems = 1 << 15;
    label sizeOfData = vectors.sizeOfSystem * vectors.numOfSystems * sizeof(scalar);

    scalar* resouces_scalar = NULL;
    label* resouces_label = NULL;

    cudaError_t err1 = cudaMalloc(&resouces_scalar, 
        (
        kMaxx_ * vectors.sizeOfSystem +                    // table_
        vectors.sizeOfSystem +                             // dfdx_
        vectors.sizeOfSystem * vectors.sizeOfSystem +      // dfdy_
        vectors.sizeOfSystem * vectors.sizeOfSystem +      // a_
        vectors.sizeOfSystem +                             // dxOpt_
        vectors.sizeOfSystem +                             // temp_
        vectors.sizeOfSystem +                             // y0_
        vectors.sizeOfSystem +                             // ySequence_
        vectors.sizeOfSystem +                             // scale_
        vectors.sizeOfSystem +                             // dy_
        vectors.sizeOfSystem +                             // yTemp_
        vectors.sizeOfSystem                               // dydx_
    ) * vectors.numOfSystems*sizeof(scalar));

    cudaMalloc(&resouces_label, vectors.sizeOfSystem * vectors.numOfSystems*sizeof(label));
    cudaMemset(resouces_label, 0, vectors.sizeOfSystem * vectors.numOfSystems*sizeof(label));

    auto start_alloc = std::chrono::high_resolution_clock::now();
    // cudaMallocManaged(&vectors.data, sizeOfData);
    cudaMallocHost(&vectors.data, sizeOfData);
    cudaMalloc(&dev_data, sizeOfData);
    auto end_alloc = std::chrono::high_resolution_clock::now();

    auto duration_alloc = std::chrono::duration_cast<std::chrono::microseconds>(end_alloc - start_alloc);
    std::cout << "Время выделения памяти: " << duration_alloc.count() << " мкс" << std::endl;

    auto start_init = std::chrono::high_resolution_clock::now();
    init(&vectors);
    cudaMemcpy(dev_data, vectors.data, sizeOfData, cudaMemcpyDefault);
    auto end_init = std::chrono::high_resolution_clock::now();
    auto duration_init = std::chrono::duration_cast<std::chrono::microseconds>(end_init - start_init);
    std::cout << "Время инициализации: " << duration_init.count() << " мкс" << std::endl;

    std::cout << std::endl;
    for (label j=0; j < vectors.sizeOfSystem; j++)
    {
        std::cout << vectors.data[0 * vectors.sizeOfSystem + j] << " ";
    } std::cout << std::endl; std::cout << std::endl;

    label threads = 256;
    label blocks = cuda::ceil_div(vectors.numOfSystems, threads);
    
    cudaEvent_t start_kernel, stop_kernel;
    cudaEventCreate(&start_kernel);
    cudaEventCreate(&stop_kernel);
    
    cudaEventRecord(start_kernel);
    seulex_solve<<<blocks, threads>>>(vectors.data, vectors.numOfSystems, step, xEnd, resouces_scalar, resouces_label);
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

    for (label i=0; i < 5; ++i){
    for (label j=0; j < vectors.sizeOfSystem; j++)
    {
        std::cout << vectors.data[i * vectors.sizeOfSystem + j] << " ";
    } std::cout << std::endl;
    } std::cout << std::endl;

    std::cout << "0.000737131 0.000144249 5.88873e-05 0.00117565 0.00238636 0.00623897 0.00285 0.00285" <<std::endl;
    
    
    auto start_free = std::chrono::high_resolution_clock::now();
    cudaFreeHost(vectors.data);
    cudaFree(dev_data);
    cudaFree(resouces_scalar);
    cudaFree(resouces_label);
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
    for (label i=0; i < vectors -> numOfSystems; ++i)
    {
        for (label j=0; j<vectors -> sizeOfSystem; ++j)
        {
            vectors -> data[i * vectors -> sizeOfSystem+ j] = 0;
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
    label sizeOfSystem_ = 8;
    
    // df/dx = 0 for autonomous system
    for (label i = 0; i < sizeOfSystem_; ++i)
    {
        dfdx[i] = 0.0;
    }
    
    // Initialize Jacobian matrix with zeros
    for (label i = 0; i < sizeOfSystem_; ++i)
    {
        for (label j = 0; j < sizeOfSystem_; ++j)
        {
            dfdy[i*sizeOfSystem_ + j] = 0.0;
        }
    }
    
    scalar y6 = y[5];
    scalar y8 = y[7];
    
    // Row 0: derivatives of y1'
    dfdy[0*sizeOfSystem_ + 0] = -1.71;
    dfdy[0*sizeOfSystem_ + 1] = 0.43;
    dfdy[0*sizeOfSystem_ + 2] = 8.32;

    // Row 1: derivatives of y2'
    dfdy[1*sizeOfSystem_ + 0] = 1.71;
    dfdy[1*sizeOfSystem_ + 1] = -8.75;

    // Row 2: derivatives of y3'
    dfdy[2*sizeOfSystem_ + 2] = -10.03;
    dfdy[2*sizeOfSystem_ + 3] = 0.43;
    dfdy[2*sizeOfSystem_ + 4] = 0.035;

    // Row 3: derivatives of y4'
    dfdy[3*sizeOfSystem_ + 1] = 8.32;
    dfdy[3*sizeOfSystem_ + 2] = 1.71;
    dfdy[3*sizeOfSystem_ + 3] = -1.12;

    // Row 4: derivatives of y5'
    dfdy[4*sizeOfSystem_ + 4] = -1.745;
    dfdy[4*sizeOfSystem_ + 5] = 0.43;
    dfdy[4*sizeOfSystem_ + 6] = 0.43;

    // Row 5: derivatives of y6'
    dfdy[5*sizeOfSystem_ + 3] = 0.69;
    dfdy[5*sizeOfSystem_ + 4] = 1.71;
    dfdy[5*sizeOfSystem_ + 5] = -280.0 * y8 - 0.43;
    dfdy[5*sizeOfSystem_ + 6] = 0.69;
    dfdy[5*sizeOfSystem_ + 7] = -280.0 * y6;

    // Row 6: derivatives of y7'
    dfdy[6*sizeOfSystem_ + 5] = 280.0 * y8;
    dfdy[6*sizeOfSystem_ + 6] = -1.81;
    dfdy[6*sizeOfSystem_ + 7] = 280.0 * y6;

    // Row 7: derivatives of y8'
    dfdy[7*sizeOfSystem_ + 5] = -280 * y8;
    dfdy[7*sizeOfSystem_ + 6] = 1.81;
    dfdy[7*sizeOfSystem_ + 7] = -280 * y6;
}


__device__
bool seul (
    const scalar x0,
    const scalar* y0,
    const scalar dxTot,
    const label k,
    scalar* y,
    const scalar* scale,
    scalar* a_,
    scalar* dfdy_,
    label* pivotIndices_,
    scalar* dy_,
    scalar* yTemp_,
    scalar* dydx_,
    scalar theta_
)
{
    label nSteps = nSeq_[k];
    scalar dx = dxTot/nSteps;
    
    for (label i=0; i<sizeOfSystem_; i++)
    {
        for (label j=0; j<sizeOfSystem_; j++)
        {
            a_[i*sizeOfSystem_ + j] = -dfdy_[i*sizeOfSystem_ + j];
        }
        a_[i*sizeOfSystem_ + i] += 1/dx;
    }

    LUDecompose(a_, pivotIndices_, sizeOfSystem_);

    scalar xnew = x0 + dx;
    derivatives(xnew, y0, dy_);
    LUBacksubstitute(a_, pivotIndices_, dy_, sizeOfSystem_);

    for(label i=0; i<sizeOfSystem_; ++i)
    {
        yTemp_[i] = y0[i];
    }

    for (label nn=1; nn<nSteps; nn++)
    {
        for(label i=0; i<sizeOfSystem_; ++i)
        {
            yTemp_[i] += dy_[i];
        }

        xnew += dx;

        if (nn == 1 && k<=1)
        {
            scalar dy1 = 0;
            for (label i=0; i<sizeOfSystem_; i++)
            {
                dy1 += sqr(dy_[i]/scale[i]);
            }
            dy1 = sqrt(dy1);

            derivatives(x0 + dx, yTemp_, dydx_);
            for (label i=0; i<sizeOfSystem_; i++)
            {
                dy_[i] = dydx_[i] - dy_[i]/dx;
            }

            LUBacksubstitute(a_, pivotIndices_, dy_, sizeOfSystem_);

            const scalar denom = min(1.0, dy1 + SMALL);
            scalar dy2 = 0;
            for (label i=0; i<sizeOfSystem_; i++)
            {
                // Test of dy_[i] to avoid overflow
                if (fabs(dy_[i]) > scale[i]*denom)
                {
                    theta_ = 1;
                    return false;
                }

                dy2 += sqr(dy_[i]/scale[i]);
            }
            dy2 = sqrt(dy2);
            theta_ = dy2/denom;

            if (theta_ > 1)
            {
                return false;
            }
        }

        derivatives(xnew, yTemp_, dy_);
        LUBacksubstitute(a_, pivotIndices_, dy_, sizeOfSystem_);
    }

    sumVec(y, yTemp_, dy_, sizeOfSystem_);

    return true;
}


__global__
void seulex_solve(scalar* data, label numOfSystems, stepState step, scalar xEnd, scalar* resouces_scalar, label* resouces_label)
{
    label workIndex = threadIdx.x + blockIdx.x*blockDim.x;

    if (workIndex < numOfSystems)
    {
        scalar theta_, logTol;
        label kTarg_;

        label resouces_scalar_size = kMaxx_ * sizeOfSystem_ +    // table_
            sizeOfSystem_ +                                      // dfdx_
            sizeOfSystem_ * sizeOfSystem_ +                      // dfdy_
            sizeOfSystem_ * sizeOfSystem_ +                      // a_
            sizeOfSystem_ +                                      // dxOpt_
            sizeOfSystem_ +                                      // temp_
            sizeOfSystem_ +                                      // y0_
            sizeOfSystem_ +                                      // ySequence_
            sizeOfSystem_ +                                      // scale_
            sizeOfSystem_ +                                      // dy_
            sizeOfSystem_ +                                      // yTemp_
            sizeOfSystem_;                                       // dydx_

        scalar* table_  = (resouces_scalar + workIndex * resouces_scalar_size);
        scalar* dfdx_   = (table_ + kMaxx_ * sizeOfSystem_);
        scalar* dfdy_   = (dfdx_ + sizeOfSystem_);
        scalar* a_      = (dfdy_ + sizeOfSystem_ * sizeOfSystem_);
        label* pivotIndices_ = (label*)resouces_label + workIndex * sizeOfSystem_;

        scalar* dxOpt_  = (a_ + sizeOfSystem_ * sizeOfSystem_);
        scalar* temp_   = (dxOpt_ + sizeOfSystem_); 
        scalar* y0_     = (temp_ + sizeOfSystem_);
        scalar* ySequence_  = (y0_ + sizeOfSystem_);
        scalar* scale_  = (ySequence_ + sizeOfSystem_); 

        scalar* dy_     = (scale_ + sizeOfSystem_);
        scalar* yTemp_  = (dy_ + sizeOfSystem_); 
        scalar* dydx_   = (yTemp_ + sizeOfSystem_);

        scalar x = 0;
        scalar dx = xEnd;
        scalar* y   = (data + workIndex * sizeOfSystem_);

        do
        {
            
            temp_[0] = GREAT;
            dx = step.dxTry;

            copyVec(y0_, y, sizeOfSystem_);

            dxOpt_[0] = fabs(0.1*dx);

            if (step.first || step.prevReject)
            {
                theta_ = 2*jacRedo_;
            }

            if (step.first)
            {
                logTol = -log10(relTol_ + absTol_)*0.6 + 0.5;
                kTarg_ = max(1, min(kMaxx_ - 1, label(logTol)));
            }

            for (label i=0; i < sizeOfSystem_; ++i)
            {
                scale_[i] = absTol_ + relTol_*fabs(y[i]);
            }


            bool jacUpdated = false;

            if (theta_ > jacRedo_)
            {
                jacobian(x, y, dfdx_, dfdy_);
                jacUpdated = true;
            }

            label k;
            scalar dxNew = fabs(dx);
            bool firstk = true;

            

            while (firstk || step.reject)
            {
                dx = step.forward ? dxNew : -dxNew;
                firstk = false;
                step.reject = false;

                scalar errOld = 0;

                for (k=0; k<=kTarg_+1; k++)
                {
                    bool success = seul(
                        x, 
                        y0_, 
                        dx, 
                        k, 
                        ySequence_, 
                        scale_, 
                        a_, 
                        dfdy_,
                        pivotIndices_,
                        dy_,
                        yTemp_,
                        dydx_,
                        theta_
                    );

                    if (!success)
                    {
                        step.reject = true;
                        dxNew = fabs(dx)*stepFactor5_;
                        break;
                    }

                    if (k == 0)
                    {
                        copyVec(y, ySequence_, sizeOfSystem_);
                    }
                    else
                    {
                        for (label i=0; i<sizeOfSystem_; ++i)
                        {
                            table_[(k-1) * sizeOfSystem_ + i] = ySequence_[i];
                        }
                    }

                    if (k != 0)
                    {
                        extrapolate(k, sizeOfSystem_, table_, y);
                        scalar err = 0;
                        for (label i=0; i<sizeOfSystem_; ++i)
                        {
                            scale_[i] = absTol_ + relTol_*fabs(y0_[i]);
                            err += sqr((y[i] - table_[i])/scale_[i]);
                        }
                        err = sqrt(err/sizeOfSystem_);
                        if (err > 1/SMALL || (k > 1 && err >= errOld))
                        {
                            step.reject = true;
                            dxNew = fabs(dx)*stepFactor5_;
                            break;
                        }
                        errOld = min(4*err, 1.0);
                        scalar expo = 1.0/(k + 1);
                        scalar facmin = pow(stepFactor3_, expo);
                        scalar fac;
                        if (err == 0)
                        {
                            fac = 1/facmin;
                        }
                        else
                        {
                            fac = stepFactor2_/pow(err/stepFactor1_, expo);
                            fac = max(facmin/stepFactor4_, min(1/facmin, fac));
                        }
                        dxOpt_[k] = fabs(dx*fac);
                        temp_[k] = cpu_[k]/dxOpt_[k];

                        if ((step.first || step.last) && err <= 1)
                        {
                            break;
                        }

                        if
                        (
                            k == kTarg_ - 1
                        && !step.prevReject
                        && !step.first && !step.last
                        )
                        {
                            if (err <= 1)
                            {
                                break;
                            }
                            else if (err > nSeq_[kTarg_]*nSeq_[kTarg_ + 1]*4)
                            {
                                step.reject = true;
                                kTarg_ = k;
                                if (kTarg_>1 && temp_[k-1] < kFactor1_*temp_[k])
                                {
                                    kTarg_--;
                                }
                                dxNew = dxOpt_[kTarg_];
                                break;
                            }
                        }

                        if (k == kTarg_)
                        {
                            if (err <= 1)
                            {
                                break;
                            }
                            else if (err > nSeq_[k + 1]*2)
                            {
                                step.reject = true;
                                if (kTarg_>1 && temp_[k-1] < kFactor1_*temp_[k])
                                {
                                    kTarg_--;
                                }
                                dxNew = dxOpt_[kTarg_];
                                break;
                            }
                        }

                        if (k == kTarg_+1)
                        {
                            if (err > 1)
                            {
                                step.reject = true;
                                if
                                (
                                    kTarg_ > 1
                                && temp_[kTarg_-1] < kFactor1_*temp_[kTarg_]
                                )
                                {
                                    kTarg_--;
                                }
                                dxNew = dxOpt_[kTarg_];
                            }
                            break;
                        }
                    }
                }
                if (step.reject)
                {
                    step.prevReject = true;
                    if (!jacUpdated)
                    {
                        theta_ = 2*jacRedo_;

                        if (theta_ > jacRedo_ && !jacUpdated)
                        {
                            jacobian(x, y, dfdx_, dfdy_);
                            jacUpdated = true;
                        }
                    }
                }
            }
            
            jacUpdated = false;

            step.dxDid = dx;
            x += dx;

            label kopt;
            if (k == 1)
            {
                kopt = 2;
            }
            else if (k <= kTarg_)
            {
                kopt=k;
                if (temp_[k-1] < kFactor1_*temp_[k])
                {
                    kopt = k - 1;
                }
                else if (temp_[k] < kFactor2_*temp_[k - 1])
                {
                    kopt = min(k + 1, kMaxx_ - 1);
                }
            }
            else
            {
                kopt = k - 1;
                if (k > 2 && temp_[k-2] < kFactor1_*temp_[k - 1])
                {
                    kopt = k - 2;
                }
                if (temp_[k] < kFactor2_*temp_[kopt])
                {
                    kopt = min(k, kMaxx_ - 1);
                }
            }

            if (step.prevReject)
            {
                kTarg_ = min(kopt, k);
                dxNew = min(fabs(dx), dxOpt_[kTarg_]);
                step.prevReject = false;
            }
            else
            {
                if (kopt <= k)
                {
                    dxNew = dxOpt_[kopt];
                }
                else
                {
                    if (k < kTarg_ && temp_[k] < kFactor2_*temp_[k - 1])
                    {
                        dxNew = dxOpt_[k]*cpu_[kopt + 1]/cpu_[k];
                    }
                    else
                    {
                        dxNew = dxOpt_[k]*cpu_[kopt]/cpu_[k];
                    }
                }
                kTarg_ = kopt;
            }

            step.dxTry = step.forward ? dxNew : -dxNew;
        } 
        while (x < xEnd);

    }
}


