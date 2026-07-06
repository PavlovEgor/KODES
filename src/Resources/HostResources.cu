#include "HostResources.cuh"

namespace kodes 
{

HostResources::HostResources(const label numOfSystems, const label sizeOfSystem, const label numOfParameters)
    : Resources(numOfSystems, sizeOfSystem, numOfParameters)
{
    this->vectors       = (scalar**)malloc(sizeOfSystem * sizeof(scalar*));
    this->parameters    = (scalar**)malloc(numOfParameters * sizeof(scalar*));
}

HostResources::~HostResources()
{
    free(this->vectors);
    free(this->parameters);
}

void HostResources::printVectori(const label i) const
{
    for (label j = 0; j < sizeOfSystem_; ++j) {
        printf("%0.5f ", this->vectors[j][i]);
    }
    printf("\n");
}

void HostResources::printParameteri(const label i) const
{
    for (label j = 0; j < sizeOfSystem_; ++j) {
        printf("%f ", this->parameters[j][i]);
    }
    printf("\n");
}

}