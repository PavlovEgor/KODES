#include "SeulexConstants.cuh"

__constant__ scalar stepFactor1_ = 0.6;
__constant__ scalar stepFactor2_ = 0.93;
__constant__ scalar stepFactor3_ = 0.1;
__constant__ scalar stepFactor4_ = 4;
__constant__ scalar stepFactor5_ = 0.5;
__constant__ scalar kFactor1_ = 0.7;
__constant__ scalar kFactor2_ = 0.9;

__constant__ label nSeq_[iMaxx_];
__constant__ scalar gpu_[iMaxx_];
__constant__ scalar coeff_[iMaxx_ * iMaxx_];

__host__ void kodes::uploadSeulexConstants()
{
    // The gpu time factors for the major parts of the algorithm
    const scalar gpuFunc = 2, gpuJac = 40, gpuLU = 17, gpuSolve = 1;

    label host_nSeq_[iMaxx_];
    scalar host_gpu_[iMaxx_];
    scalar host_coeff_[iMaxx_ * iMaxx_];

    host_nSeq_[0] = 2;
    host_nSeq_[1] = 3;

    for (int i=2; i<iMaxx_; i++)
    {
        host_nSeq_[i] = 2 * host_nSeq_[i-2];
    }
    host_gpu_[0] = gpuJac + gpuLU + host_nSeq_[0]*(gpuFunc + gpuSolve);

    for (int k=0; k<kMaxx_; k++)
    {
        host_gpu_[k+1] = host_gpu_[k] + (host_nSeq_[k+1]-1)*(gpuFunc + gpuSolve) + gpuLU;
    }

    // Set the extrapolation coefficients array
    for (int k=0; k<iMaxx_; k++)
    {
        for (int l=0; l<k; l++)
        {
            scalar ratio = scalar(host_nSeq_[k])/host_nSeq_[l];
            host_coeff_[k + l*iMaxx_] = 1/(ratio - 1);
        }
    }

    CUDA_CHECK(cudaMemcpyToSymbol(nSeq_, host_nSeq_, iMaxx_ * sizeof(label)));
    CUDA_CHECK(cudaMemcpyToSymbol(gpu_, host_gpu_, iMaxx_ * sizeof(scalar)));
    CUDA_CHECK(cudaMemcpyToSymbol(coeff_, host_coeff_, iMaxx_ * iMaxx_ * sizeof(scalar)));
}
