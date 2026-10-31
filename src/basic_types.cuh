#ifndef KODES_BASIC_TYPES
#define KODES_BASIC_TYPES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <type_traits>

namespace kodes
{
    using scalar = double;
    using label  = int;
}

#define SMALL 1.0e-15
#define GREAT 1.0e+15

#define GRID_DIM (blockDim.x * gridDim.x)
#define T_ID (threadIdx.x + blockIdx.x * blockDim.x)

// scratch space: the calling thread's slot, one per resident thread
#define INDEXVEC(i) (T_ID + (i) * GRID_DIM)
#define INDEXMAT(i, j, size) (T_ID + ((i) + (j) * (size)) * GRID_DIM)

// state space: system `system` of the batch, one slot per system
#define INDEXSTATE(system, i, batchSize) ((system) + (i) * (batchSize))

#define KODES_BLOCK_SIZE 256

#define CUDA_CHECK(call)                                                          \
    do {                                                                         \
        cudaError_t err__ = (call);                                              \
        if (err__ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(err__));                                  \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                        \
    } while (0)

#define CUDA_CHECK_LAST()                                                         \
    do {                                                                         \
        cudaError_t err__ = cudaGetLastError();                                  \
        if (err__ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA kernel launch error at %s:%d: %s\n", __FILE__, \
                    __LINE__, cudaGetErrorString(err__));                        \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                        \
    } while (0)

namespace kodes
{
    __host__ __device__ inline label blockSize(const label ensembleSize)
    {
        return ensembleSize < KODES_BLOCK_SIZE ? ensembleSize : KODES_BLOCK_SIZE;
    }

    __host__ __device__ inline label numOfBlocks(const label ensembleSize)
    {
        const label threads = blockSize(ensembleSize);
        return (ensembleSize + threads - 1) / threads;
    }

    // Dynamic shared memory of one block: the 4*blockDim.x scratch doubles
    // pyJac's generated kernels expect (shared_temp), and behind them any
    // scratch a block shares, when the plan put it there - which in this
    // version it never does, see LaunchConfig::blockScratchInShared.
    __host__ __device__ inline size_t
    sharedMemorySize(const label threads, const size_t blockScratchBytes = 0)
    {
        return size_t(4 * threads) * sizeof(scalar) + blockScratchBytes;
    }
}

// Views of device memory, which every buffer of the library is held and
// passed as.
//
// A view owns nothing. Copying one, or passing it by value, gives another name
// for the same memory, as copying a pointer does. Assigning to one writes the
// elements, as assigning through a reference does: `y = yTemp` copies the
// vector, `y = yTemp + dy` sums into it, neither repoints y. A view there is
// nothing sensible to assign to - a matrix, the batch - cannot be assigned at
// all, so that nobody repoints one by accident either.
//
// A default constructed view is left alone, which is what the device objects
// need: their constructors run after the stub has been byte-copied to the
// device, and must not overwrite what the copy brought (see
// Factory/device_object.cuh). allocate() and deallocate() are called on that
// stub, on the host.
namespace kodes
{
    template <typename T> struct ThreadVector;

    // a + b, evaluated element by element when it is assigned to a vector
    template <typename T>
    struct ThreadVectorSum
    {
        ThreadVector<const T> a;
        ThreadVector<const T> b;

        __device__ T operator[](const label i) const
        {
            return a[i] + b[i];
        }
    };

    // The calling thread's own vector in scratch space: element i is at
    // INDEXVEC(i)
    template <typename T>
    struct ThreadVector
    {
        using value_type = std::remove_const_t<T>;

        T* data;
        label size;

        ThreadVector() = default;

        ThreadVector(const ThreadVector&) = default;

        __host__ __device__ ThreadVector(T* data, const label size)
            : data(data), size(size)
        {}

        // a read only view of a vector that may be written
        template <typename U, typename = std::enable_if_t<std::is_same_v<const U, T>>>
        __host__ __device__ ThreadVector(const ThreadVector<U>& v)
            : data(v.data), size(v.size)
        {}

        __device__ T& operator[](const label i) const
        {
            return data[INDEXVEC(i)];
        }

        __device__ const ThreadVector& operator=(const ThreadVector& v) const
        {
            return assign(v);
        }

        template <typename U>
        __device__ const ThreadVector& operator=(const ThreadVector<U>& v) const
        {
            return assign(v);
        }

        __device__ const ThreadVector& operator=(const ThreadVectorSum<value_type>& v) const
        {
            return assign(v);
        }

        template <typename U>
        __device__ const ThreadVector& operator+=(const ThreadVector<U>& v) const
        {
            for (label i = 0; i < size; ++i)
            {
                (*this)[i] += v[i];
            }
            return *this;
        }

        // `n` elements per resident thread
        __host__ void allocate(const label n, const label scratchSize)
        {
            size = n;
            CUDA_CHECK(cudaMalloc(&data, size_t(n) * scratchSize * sizeof(T)));
        }

        __host__ void deallocate()
        {
            CUDA_CHECK(cudaFree(data));
        }

    private:

        template <typename V>
        __device__ const ThreadVector& assign(const V& v) const
        {
            for (label i = 0; i < size; ++i)
            {
                (*this)[i] = v[i];
            }
            return *this;
        }
    };

    template <typename A, typename B>
    __device__ ThreadVectorSum<std::remove_const_t<A>>
    operator+(const ThreadVector<A>& a, const ThreadVector<B>& b)
    {
        static_assert
        (
            std::is_same_v<std::remove_const_t<A>, std::remove_const_t<B>>,
            "a sum of vectors of one type"
        );
        return {a, b};
    }

    // The calling thread's own size x size matrix in scratch space: element
    // (i, j) is at INDEXMAT(i, j, size)
    template <typename T>
    struct ThreadMatrix
    {
        T* data;
        label size;

        ThreadMatrix() = default;

        ThreadMatrix(const ThreadMatrix&) = default;

        ThreadMatrix& operator=(const ThreadMatrix&) = delete;

        __host__ __device__ ThreadMatrix(T* data, const label size)
            : data(data), size(size)
        {}

        template <typename U, typename = std::enable_if_t<std::is_same_v<const U, T>>>
        __host__ __device__ ThreadMatrix(const ThreadMatrix<U>& m)
            : data(m.data), size(m.size)
        {}

        __device__ T& operator()(const label i, const label j) const
        {
            return data[INDEXMAT(i, j, size)];
        }

        // `n` x `n` elements per resident thread
        __host__ void allocate(const label n, const label scratchSize)
        {
            size = n;
            CUDA_CHECK(cudaMalloc(&data, size_t(n) * n * scratchSize * sizeof(T)));
        }

        __host__ void deallocate()
        {
            CUDA_CHECK(cudaFree(data));
        }
    };

    // One value per resident thread, the calling thread's at INDEXVEC(0). It
    // stands for that value: it is read where a T is expected and written by
    // assigning to it, and get() names it where a T& is wanted - printf()
    // included, which takes no class.
    template <typename T>
    struct ThreadValue
    {
        T* data;

        __device__ T& get() const
        {
            return data[INDEXVEC(0)];
        }

        __device__ operator T&() const
        {
            return get();
        }

        __device__ const ThreadValue& operator=(const T& value) const
        {
            get() = value;
            return *this;
        }

        __device__ const ThreadValue& operator=(const ThreadValue& v) const
        {
            get() = v.get();
            return *this;
        }

        __host__ void allocate(const label scratchSize)
        {
            CUDA_CHECK(cudaMalloc(&data, size_t(scratchSize) * sizeof(T)));
        }

        __host__ void deallocate()
        {
            CUDA_CHECK(cudaFree(data));
        }
    };

    // The whole batch in state space, `size` components of batchSize systems:
    // component i of a system is at INDEXSTATE(system, i, batchSize)
    template <typename T>
    struct BatchArray
    {
        T* data;
        label size;
        label batchSize;

        BatchArray() = default;

        BatchArray(const BatchArray&) = default;

        BatchArray& operator=(const BatchArray&) = delete;

        __device__ T& operator()(const label system, const label i) const
        {
            return data[INDEXSTATE(system, i, batchSize)];
        }

        // Where component i of the whole batch starts, for the host to copy a
        // batch in and out with cudaMemcpy
        __host__ T* component(const label i) const
        {
            return data + size_t(i) * batchSize;
        }

        __host__ void allocate(const label n, const label systems)
        {
            size = n;
            batchSize = systems;
            CUDA_CHECK(cudaMalloc(&data, size_t(n) * systems * sizeof(T)));
        }

        __host__ void deallocate()
        {
            CUDA_CHECK(cudaFree(data));
        }
    };
}


#endif
