#include "Balancer.cuh"

namespace kodes
{

__global__ void
computeKeys(Balancer* balancer, const DeviceResources* resources, const label realBatchSize)
{
    scalar* __restrict__ keys = balancer->keys();

    for (label system = T_ID; system < realBatchSize; system += GRID_DIM)
    {
        keys[system] = balancer->key(resources, system);
    }
}

}

__host__ void
kodes::Balancer::allocate(const label batchSize)
{
    CUDA_CHECK(cudaMalloc(&keys_, size_t(batchSize) * sizeof(scalar)));
    CUDA_CHECK(cudaMalloc(&order_, size_t(batchSize) * sizeof(label)));

    hostKeys_ = (scalar*)malloc(size_t(batchSize) * sizeof(scalar));
    hostOrder_ = (label*)malloc(size_t(batchSize) * sizeof(label));

    if (!hostKeys_ || !hostOrder_)
    {
        fprintf(stderr, "Balancer::allocate error at %s:%d: out of host memory\n", __FILE__, __LINE__);
        std::exit(EXIT_FAILURE);
    }
}

__host__ void
kodes::Balancer::deallocate()
{
    CUDA_CHECK(cudaFree(keys_));
    CUDA_CHECK(cudaFree(order_));

    free(hostKeys_);
    free(hostOrder_);
}

__host__ void
kodes::Balancer::balance
(
    Balancer* devBalancer,
    DeviceResources* resources,
    const label realBatchSize,
    const LaunchConfig& config
)
{
    if (realBatchSize <= 0 || realBatchSize > batchSize_)
    {
        fprintf(stderr, "Balancer::balance error at %s:%d: realBatchSize out of range\n", __FILE__, __LINE__);
        std::exit(EXIT_FAILURE);
    }

    kodes::computeKeys<<<config.blocks, config.threads>>>(devBalancer, resources, realBatchSize);
    CUDA_CHECK_LAST();

    CUDA_CHECK(cudaMemcpy(hostKeys_, keys_, size_t(realBatchSize) * sizeof(scalar), cudaMemcpyDeviceToHost));

    for (label i = 0; i < realBatchSize; ++i)
    {
        hostOrder_[i] = i;
    }

    quickSortByKey(hostKeys_, hostOrder_, realBatchSize);

    CUDA_CHECK(cudaMemcpy(order_, hostOrder_, size_t(realBatchSize) * sizeof(label), cudaMemcpyHostToDevice));
}

namespace
{

inline void swapItems(scalar* keys, label* order, const label i, const label j)
{
    const scalar key = keys[i];
    keys[i] = keys[j];
    keys[j] = key;

    const label index = order[i];
    order[i] = order[j];
    order[j] = index;
}

// Ranges shorter than this are left to the final insertion pass
const label insertionLimit = 16;

// Sorting a range never needs more than log2(size) entries, because the larger
// half is pushed and the smaller one is looped on
const label maxDepth = 64;

}

__host__ void
kodes::quickSortByKey(scalar* keys, label* order, const label size)
{
    if (size < 2)
    {
        return;
    }

    struct Range { label first; label last; };

    Range stack[maxDepth];
    label top = 0;

    stack[top++] = {0, size - 1};

    while (top > 0)
    {
        Range range = stack[--top];

        while (range.last - range.first >= insertionLimit)
        {
            // median of the first, middle and last element, parked at `first`
            const label middle = range.first + (range.last - range.first) / 2;

            if (keys[middle] < keys[range.first])   swapItems(keys, order, middle, range.first);
            if (keys[range.last] < keys[range.first]) swapItems(keys, order, range.last, range.first);
            if (keys[range.last] < keys[middle])    swapItems(keys, order, range.last, middle);

            swapItems(keys, order, middle, range.first);

            const scalar pivot = keys[range.first];

            label i = range.first - 1;
            label j = range.last + 1;

            while (true)
            {
                do { ++i; } while (keys[i] < pivot);
                do { --j; } while (keys[j] > pivot);

                if (i >= j)
                {
                    break;
                }

                swapItems(keys, order, i, j);
            }

            // recurse into the smaller half, loop on the larger one
            if (j - range.first < range.last - j)
            {
                if (top == maxDepth)
                {
                    fprintf(stderr, "kodes::quickSortByKey error at %s:%d: stack overflow\n", __FILE__, __LINE__);
                    std::exit(EXIT_FAILURE);
                }
                stack[top++] = {label(j + 1), range.last};
                range.last = j;
            }
            else
            {
                if (top == maxDepth)
                {
                    fprintf(stderr, "kodes::quickSortByKey error at %s:%d: stack overflow\n", __FILE__, __LINE__);
                    std::exit(EXIT_FAILURE);
                }
                stack[top++] = {range.first, j};
                range.first = label(j + 1);
            }
        }

        for (label i = range.first + 1; i <= range.last; ++i)
        {
            const scalar key = keys[i];
            const label index = order[i];

            label j = i - 1;
            while (j >= range.first && keys[j] > key)
            {
                keys[j + 1] = keys[j];
                order[j + 1] = order[j];
                --j;
            }

            keys[j + 1] = key;
            order[j + 1] = index;
        }
    }
}
