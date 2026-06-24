// HIRESSystem.cpp
#include "HIRESSystem.cuh"

__global__ void 
constructGPU(kodes::HIRESSystem* system, const label sizeOfSystem)
{
    new (system) kodes::HIRESSystem(sizeOfSystem);
}

__global__ void 
destructGPU(kodes::HIRESSystem* system) {
    delete system;
}

__host__  kodes::HIRESSystem* 
kodes::HIRESSystem::createGPU(const label sizeOfSystem) {
    HIRESSystem* ptr;
    cudaMalloc(&ptr, sizeof(HIRESSystem));
    constructGPU<<<1, 1>>>(ptr, sizeOfSystem);
    cudaDeviceSynchronize();
    return ptr;
}

__host__  void
kodes::HIRESSystem::destroyGPU(kodes::HIRESSystem* system) {
    if (system) {
        destructGPU<<<1, 1>>>(system);
        cudaDeviceSynchronize();
        cudaFree(system);
    }
}

__device__
void kodes::ChemSystem::omega(const scalar* c, const scalar T, const scalar p, scalar* dcdt) const
{
}

__device__
void kodes::ChemSystem::omegaI(const scalar* c, const scalar T, const scalar p, scalar* dcdt) const
{
}

scalar kodes::ChemSystem::omega
(
    const Reaction* R,
    const scalar* c,
    const scalar T,
    const scalar p,
    scalar* pf,
    scalar* cf,
    label* lRef,
    scalar* pr,
    scalar* cr,
    label* rRef
) const
{
    const scalar kf = R.kf(p, T, c);
    const scalar kr = R.kr(kf, p, T, c);

    pf = 1.0;
    pr = 1.0;

    const label Nl = R.lhs().size();
    const label Nr = R.rhs().size();

    label slRef = 0;
    lRef = R.lhs()[slRef].index;

    pf = kf;
    for (label s = 1; s < Nl; s++)
    {
        const label si = R.lhs()[s].index;

        if (c[si] < c[lRef])
        {
            const scalar exp = R.lhs()[slRef].exponent;
            pf *= pow(max(c[lRef], 0.0), exp);
            lRef = si;
            slRef = s;
        }
        else
        {
            const scalar exp = R.lhs()[s].exponent;
            pf *= pow(max(c[si], 0.0), exp);
        }
    }
    cf = max(c[lRef], 0.0);

    {
        const scalar exp = R.lhs()[slRef].exponent;
        if (exp < 1.0)
        {
            if (cf > SMALL)
            {
                pf *= pow(cf, exp - 1.0);
            }
            else
            {
                pf = 0.0;
            }
        }
        else
        {
            pf *= pow(cf, exp - 1.0);
        }
    }

    label srRef = 0;
    rRef = R.rhs()[srRef].index;

    // Find the matrix element and element position for the rhs
    pr = kr;
    for (label s = 1; s < Nr; s++)
    {
        const label si = R.rhs()[s].index;
        if (c[si] < c[rRef])
        {
            const scalar exp = R.rhs()[srRef].exponent;
            pr *= pow(max(c[rRef], 0.0), exp);
            rRef = si;
            srRef = s;
        }
        else
        {
            const scalar exp = R.rhs()[s].exponent;
            pr *= pow(max(c[si], 0.0), exp);
        }
    }
    cr = max(c[rRef], 0.0);

    {
        const scalar exp = R.rhs()[srRef].exponent;
        if (exp < 1.0)
        {
            if (cr>SMALL)
            {
                pr *= pow(cr, exp - 1.0);
            }
            else
            {
                pr = 0.0;
            }
        }
        else
        {
            pr *= pow(cr, exp - 1.0);
        }
    }

    return pf*cf - pr*cr;
}

__device__
void kodes::ChemSystem::derivatives(const scalar time, const scalar* c, scalar* dcdt) const
{
    const scalar T = c[nSpecie_];
    const scalar p = c[nSpecie_ + 1];

    forAll(c_, i)
    {
        c_[i] = max(c[i], 0.0);
    }

    omega(c_, T, p, dcdt);

    // Constant pressure
    // dT/dt = ...
    scalar rho = 0.0;
    for (label i = 0; i < nSpecie_; i++)
    {
        const scalar W = specieThermo_[i].W();
        rho += W*c_[i];
    }
    scalar cp = 0.0;
    for (label i=0; i<nSpecie_; i++)
    {
        cp += c_[i]*specieThermo_[i].cp(p, T);
    }
    cp /= rho;

    scalar dT = 0.0;
    for (label i = 0; i < nSpecie_; i++)
    {
        const scalar hi = specieThermo_[i].ha(p, T);
        dT += hi*dcdt[i];
    }
    dT /= rho*cp;

    dcdt[nSpecie_] = -dT;

    // dp/dt = ...
    dcdt[nSpecie_ + 1] = 0.0;
}


__device__
void kodes::HIRESSystem::jacobian(const scalar time, const scalar* c, scalar* dfdt, scalar* dfdc) const
{
}
