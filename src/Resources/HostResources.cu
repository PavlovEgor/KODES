#include "HostResources.cuh"

namespace kodes 
{

HostResources::HostResources(const label ensembleSize, const label systemSize, const label parameterSize)
    : Resources(ensembleSize, systemSize, parameterSize),
      vectors{},
      parameters{}
{
    checkSystemDimensions(systemSize, parameterSize, "HostResources");
}

HostResources& HostResources::operator=(const HostResources& other)
{
    if (this == &other) {
        return *this;
    }
    
    Resources::operator=(other);
    
    for (label i = 0; i < systemSize_; ++i) {
        for (label j = 0; j < ensembleSize_; ++j) {
            this->vectors[i][j] = other.vectors[i][j];
        }
    }
    
    for (label i = 0; i < parameterSize_; ++i) {
        for (label j = 0; j < ensembleSize_; ++j) {
            this->parameters[i][j] = other.parameters[i][j];
        }
    }
    
    return *this;
}

void HostResources::printVectori(const label i) const
{
    for (label j = 0; j < systemSize_; ++j) {
        printf("%0.5f ", this->vectors[j][i]);
    }
    printf("\n");
}

void HostResources::printParameteri(const label i) const
{
    for (label j = 0; j < parameterSize_; ++j) {
        printf("%f ", this->parameters[j][i]);
    }
    printf("\n");
}

}
