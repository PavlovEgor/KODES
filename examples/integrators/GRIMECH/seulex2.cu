
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda/cmath>
#include <cuda_runtime.h>
#include "seulex2.cuh"
#include <iostream>
#include <chrono> 


int main(){

    label numOfSystems = 1 << 5;

    kodes::HostResources            host_res(numOfSystems, NSP, 1);

    set_same_initial_conditions(host_res.numOfSystems(), host_res.vectors, host_res.parameters);

    host_res.printVectori(0);

    mechanism_memory *h_mem = (mechanism_memory*)malloc(sizeof(mechanism_memory));
    mechanism_memory *d_mem = nullptr;

    initialize_gpu_memory(host_res.numOfSystems(), &h_mem, &d_mem);

    kodes::GRIMESHSystem* ode_prt = kodes::GRIMESHSystem::createGPU(d_mem);

    kodes::SeulexDeviceResources   host_res_dev(host_res.numOfSystems(), host_res.sizeOfSystem(), host_res.numOfParameters());

    kodes::SeulexDeviceResources*   res_prt = kodes::SeulexDeviceResources::create(numOfSystems, host_res.sizeOfSystem(), 1, &host_res_dev);

    kodes::Operator<kodes::HostResources, kodes::SeulexDeviceResources> op(&host_res, &host_res_dev);

    op.cpyHostToDevice();

    scalar xEnd = 1;
    stepState step(xEnd);

    label threads = host_res.numOfSystems() <= 256 ? host_res.numOfSystems() : 256;
    label blocks = cuda::ceil_div(host_res.numOfSystems(), threads);
    size_t sharedMemSize = (3 * threads + threads) * sizeof(scalar); 

    seulex_solve<<<blocks, threads, sharedMemSize>>>(ode_prt, res_prt, step);
    
    op.cpyDeviceToHost();

    host_res.printVectori(0);
    host_res.printVectori(1);

    kodes::GRIMESHSystem::destroyGPU(ode_prt);
    kodes::SeulexDeviceResources::destroy(res_prt, &host_res_dev);

    return 0;
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
    scalar theta_,
    kodes::GRIMESHSystem* ode,
    kodes::SeulexDeviceResources* res
)
{
    label nSteps = nSeq_[k];
    scalar dx = dxTot/nSteps;
    
    for (label i=0; i<sizeOfSystem_; i++)
    {
        for (label j=0; j<sizeOfSystem_; j++)
        {
            a_[INDEX(i*sizeOfSystem_ + j)] = -dfdy_[INDEX(i*sizeOfSystem_ + j)];
        }
        a_[INDEX(i*sizeOfSystem_ + i)] += 1/dx;
    }

    LUDecompose(a_, pivotIndices_, sizeOfSystem_);

    scalar xnew = x0 + dx;
    ode->derivatives(xnew, res->y0(), res->dy());
    LUBacksubstitute(a_, pivotIndices_, dy_, sizeOfSystem_);

    copyVec(yTemp_, y0, sizeOfSystem_);

    for (label nn=1; nn<nSteps; nn++)
    {
        // for(label i=0; i<sizeOfSystem_; ++i)
        // {
        //     yTemp_[i] += dy_[i];
        // }
        sumVec(yTemp_, yTemp_, dy_, sizeOfSystem_);

        xnew += dx;

        if (nn == 1 && k<=1)
        {
            scalar dy1 = 0;
            for (label i=0; i<sizeOfSystem_; i++)
            {
                dy1 += sqr(dy_[INDEX(i)]/scale[INDEX(i)]);
            }
            dy1 = sqrt(dy1);

            ode->derivatives(x0 + dx, yTemp_, dydx_);
            for (label i=0; i<sizeOfSystem_; i++)
            {
                dy_[INDEX(i)] = dydx_[INDEX(i)] - dy_[INDEX(i)]/dx;
            }

            LUBacksubstitute(a_, pivotIndices_, dy_, sizeOfSystem_);

            const scalar denom = min(1.0, dy1 + SMALL);
            scalar dy2 = 0;
            for (label i=0; i<sizeOfSystem_; i++)
            {
                // Test of dy_[i] to avoid overflow
                if (fabs(dy_[INDEX(i)]) > scale[INDEX(i)]*denom)
                {
                    theta_ = 1;
                    return false;
                }

                dy2 += sqr(dy_[INDEX(i)]/scale[INDEX(i)]);
            }
            dy2 = sqrt(dy2);
            theta_ = dy2/denom;

            if (theta_ > 1)
            {
                return false;
            }
        }

        ode->derivatives(xnew, yTemp_, dy_);
        LUBacksubstitute(a_, pivotIndices_, dy_, sizeOfSystem_);
    }

    sumVec(y, yTemp_, dy_, sizeOfSystem_);

    return true;
}


__global__
void seulex_solve(kodes::GRIMESHSystem* ode, kodes::SeulexDeviceResources* res, stepState step)
{
    label workIndex = threadIdx.x + blockIdx.x*blockDim.x;

    if (workIndex == 0)
    {
        printf("%d \n", GRID_DIM);
    }

    if (workIndex < res->numOfSystems())
    {
        scalar theta_, logTol;
        label kTarg_;


        scalar* table_ = res->table();
        scalar* dfdx_  = res->dfdx();
        scalar* dfdy_  = res->dfdy();
        scalar* a_     = res->a();
        label* pivotIndices_ = res->pivotIndices();
        
        scalar* dxOpt_ = res->dxOpt();
        scalar* temp_  = res->temp();
        scalar* y0_    = res->y0();
        scalar* ySequence_ = res->ySequence();
        scalar* scale_ = res->scale();
        
        scalar* dy_    = res->dy();
        scalar* yTemp_ = res->yTemp();
        scalar* dydx_  = res->dydx();
        scalar* y      = res->vectors;

        scalar x = 0;
        scalar xEnd = step.dxTry;
        step.dxTry /= 2;
        scalar dx = xEnd/2;

        do
        {
            if (workIndex == 0)
            {printf("New step:");
            for (label j = 0; j < sizeOfSystem_; ++j) {
                 printf("%0.2f ", y[INDEX(j)]);
            }
            printf("\n dx=%0.16f \n", dx);
            }
            temp_[INDEX(0)] = GREAT;
            dx = step.dxTry;

            copyVec(y0_, y, sizeOfSystem_);

            dxOpt_[INDEX(0)] = fabs(0.1*dx);

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
                scale_[INDEX(i)] = absTol_ + relTol_*fabs(y[INDEX(i)]);
            }

            bool jacUpdated = false;
            if (theta_ > jacRedo_)
            {
                ode->jacobian(x, y, dfdx_, dfdy_);
                jacUpdated = true;
            }

            label k;
            scalar dxNew = fabs(dx);
            bool firstk = true;


            while (firstk || step.reject)
            {
                // for (label j = 0; j < sizeOfSystem_; ++j) {
                //     printf("Sub step: %0.2f ", y[INDEX(j)]);
                // }
                // printf("\n %0.2f \n", dx);
                // }
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
                        theta_,
                        ode,
                        res
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
                            table_[INDEX((k-1) * sizeOfSystem_ + i)] = ySequence_[INDEX(i)];
                        }
                    }

                    if (k != 0)
                    {
                        extrapolate(k, sizeOfSystem_, table_, y);
                        scalar err = 0;
                        for (label i=0; i<sizeOfSystem_; ++i)
                        {
                            scale_[INDEX(i)] = absTol_ + relTol_*fabs(y0_[INDEX(i)]);
                            err += sqr((y[INDEX(i)] - table_[INDEX(i)])/scale_[INDEX(i)]);
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
                        dxOpt_[INDEX(k)] = fabs(dx*fac);
                        temp_[INDEX(k)] = cpu_[k]/dxOpt_[INDEX(k)];

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
                                if (kTarg_>1 && temp_[INDEX(k-1)] < kFactor1_*temp_[INDEX(k)])
                                {
                                    kTarg_--;
                                }
                                dxNew = dxOpt_[INDEX(kTarg_)];
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
                                if (kTarg_>1 && temp_[INDEX(k-1)] < kFactor1_*temp_[INDEX(k)])
                                {
                                    kTarg_--;
                                }
                                dxNew = dxOpt_[INDEX(kTarg_)];
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
                                && temp_[INDEX(kTarg_-1)] < kFactor1_*temp_[INDEX(kTarg_)]
                                )
                                {
                                    kTarg_--;
                                }
                                dxNew = dxOpt_[INDEX(kTarg_)];
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
                            ode->jacobian(x, y, dfdx_, dfdy_);
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
                if (temp_[INDEX(k-1)] < kFactor1_*temp_[INDEX(k)])
                {
                    kopt = k - 1;
                }
                else if (temp_[INDEX(k)] < kFactor2_*temp_[INDEX(k - 1)])
                {
                    kopt = min(k + 1, kMaxx_ - 1);
                }
            }
            else
            {
                kopt = k - 1;
                if (k > 2 && temp_[INDEX(k-2)] < kFactor1_*temp_[INDEX(k - 1)])
                {
                    kopt = k - 2;
                }
                if (temp_[INDEX(k)] < kFactor2_*temp_[INDEX(kopt)])
                {
                    kopt = min(k, kMaxx_ - 1);
                }
            }

            if (step.prevReject)
            {
                kTarg_ = min(kopt, k);
                dxNew = min(fabs(dx), dxOpt_[INDEX(kTarg_)]);
                step.prevReject = false;
            }
            else
            {
                if (kopt <= k)
                {
                    dxNew = dxOpt_[INDEX(kopt)];
                }
                else
                {
                    if (k < kTarg_ && temp_[INDEX(k)] < kFactor2_*temp_[INDEX(k - 1)])
                    {
                        dxNew = dxOpt_[INDEX(k)]*cpu_[kopt + 1]/cpu_[k];
                    }
                    else
                    {
                        dxNew = dxOpt_[INDEX(k)]*cpu_[kopt]/cpu_[k];
                    }
                }
                kTarg_ = kopt;
            }

            step.dxTry = step.forward ? dxNew : -dxNew;
        } 
        while (x < xEnd);
    }
}


