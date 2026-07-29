#pragma once

namespace kodes 
{
class kodes
{
private:

    double** ODEVectors_;
    int      numOfSystems_;
    int      sizeOfSystem_;

    RHS_Func* Func_;

    ConvTol*  convtol_;

public:
    virtual Config();
    
    virtual ~Config() = default;

    virtual void solve() = 0;

};
}
