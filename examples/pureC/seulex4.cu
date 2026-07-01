
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda/cmath>
#include <cuda_runtime.h>
#include "seulex4.cuh"
#include <iostream>
#include <chrono> 



int main(){

    label numOfSystems = 1 << 3;

    kodes::HostResources            host_res(numOfSystems, 8, 0);

    init(&host_res);

    host_res.printVectori(0);
    host_res.printVectori(1);

    kodes::HIRESSystem* ode_prt = kodes::HIRESSystem::createGPU(host_res.sizeOfSystem());

    kodes::SeulexDeviceResources   host_res_dev(host_res.numOfSystems(), host_res.sizeOfSystem(), host_res.numOfParameters());

    kodes::SeulexDeviceResources*   res_prt = kodes::SeulexDeviceResources::create(numOfSystems, host_res.sizeOfSystem(), 1, &host_res_dev);

    kodes::Operator<kodes::HostResources, kodes::SeulexDeviceResources> op(&host_res, &host_res_dev);

    op.cpyHostToDevice();

    scalar xEnd = 321.8122;
    stepState step(xEnd);

    label threads = 256;
    label blocks = cuda::ceil_div(host_res.numOfSystems(), threads);
    
    seulex_solve<<<blocks, threads>>>(ode_prt, res_prt, step);
    
    op.cpyDeviceToHost();

    host_res.printVectori(0);
    host_res.printVectori(1);

    std::cout << "0.000737131 0.000144249 5.88873e-05 0.00117565 0.00238636 0.00623897 0.00285 0.00285" <<std::endl;
    
    kodes::HIRESSystem::destroyGPU(ode_prt);
    kodes::SeulexDeviceResources::destroy(res_prt, &host_res_dev);

    return 0;
}

void init(kodes::HostResources* host_res)
{
    for (label i=0; i < host_res -> sizeOfSystem(); ++i)
    {
        host_res -> vectors[i] = (scalar*)malloc(host_res -> numOfSystems() * sizeof(scalar));
        for (label j=0; j<host_res -> numOfSystems(); ++j)
        {
            host_res -> vectors[i][j] = 0;
        }
    }
    for (label j=0; j<host_res -> numOfSystems(); ++j)
    {
        host_res -> vectors[0][j] = 1.0;
        host_res -> vectors[7][j] = 0.0057;
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
    scalar theta_,
    kodes::HIRESSystem* ode
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
    ode->derivatives(xnew, y0, dy_);
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

            ode->derivatives(x0 + dx, yTemp_, dydx_);
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

        ode->derivatives(xnew, yTemp_, dy_);
        LUBacksubstitute(a_, pivotIndices_, dy_, sizeOfSystem_);
    }

    sumVec(y, yTemp_, dy_, sizeOfSystem_);

    return true;
}


__global__
void seulex_solve(kodes::HIRESSystem* ode, kodes::SeulexDeviceResources* res, stepState step)
{
    label workIndex = threadIdx.x + blockIdx.x*blockDim.x;

    if (workIndex < res->numOfSystems())
    {
        scalar theta_, logTol;
        label kTarg_;


        scalar* table_ = res->table(workIndex);
        scalar* dfdx_  = res->dfdx(workIndex);
        scalar* dfdy_  = res->dfdy(workIndex);
        scalar* a_     = res->a(workIndex);
        label* pivotIndices_ = res->pivotIndices(workIndex);
        
        scalar* dxOpt_ = res->dxOpt(workIndex);
        scalar* temp_  = res->temp(workIndex);
        scalar* y0_    = res->y0(workIndex);
        scalar* ySequence_ = res->ySequence(workIndex);
        scalar* scale_ = res->scale(workIndex);
        
        scalar* dy_    = res->dy(workIndex);
        scalar* yTemp_ = res->yTemp(workIndex);
        scalar* dydx_  = res->dydx(workIndex);
        scalar* y      = res->y(workIndex);

        scalar x = 0;
        scalar xEnd = step.dxTry;
        scalar dx = xEnd;

        for (int i=0; i<sizeOfSystem_; ++i)
        {
            y[i] = *(res->vectors + workIndex + i * res->numOfSystems());
        }

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
                ode->jacobian(x, y, dfdx_, dfdy_);
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
                        theta_,
                        ode
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

        for (int i=0; i<sizeOfSystem_; ++i)
        {
            *(res->vectors + workIndex + i * res->numOfSystems()) = y[i];
        }
    }
}


