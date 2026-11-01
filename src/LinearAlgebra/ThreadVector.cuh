#ifndef KODES_THREAD_VECTOR
#define KODES_THREAD_VECTOR

#pragma once

#include <type_traits>

#include "basic_types.cuh"
#include "system_dimensions.cuh"
#include "DeviceBuffer.cuh"

namespace kodes
{

template <typename T, label N> class ThreadVector;

template <typename Op, typename L, typename R> class ThreadVectorExpression;

// What an element-wise expression is made of: vectors, expressions of them, and
// numbers, which stand for a vector of that number everywhere
template <typename X>
constexpr bool kIsThreadVectorOperand = false;

template <typename T, label N>
constexpr bool kIsThreadVectorOperand<ThreadVector<T, N>> = true;

template <typename Op, typename L, typename R>
constexpr bool kIsThreadVectorOperand<ThreadVectorExpression<Op, L, R>> = true;

template <typename X>
constexpr bool kIsThreadVectorScalar = std::is_arithmetic_v<X>;

// How many elements an operand has; a number has none of its own
template <typename X>
constexpr label kThreadVectorOperandSize = 0;

template <typename T, label N>
constexpr label kThreadVectorOperandSize<ThreadVector<T, N>> = N;

template <typename Op, typename L, typename R>
constexpr label kThreadVectorOperandSize<ThreadVectorExpression<Op, L, R>> =
    ThreadVectorExpression<Op, L, R>::size;

// Element i of an operand: a number is every element of itself
template <typename X>
__device__ auto element(const X& x, const label i)
{
    if constexpr (kIsThreadVectorOperand<X>)
    {
        return x[i];
    }
    else
    {
        return x;
    }
}

// The operations an expression applies to a pair of elements
struct Plus
{
    template <typename A, typename B>
    __device__ static auto apply(const A a, const B b) { return a + b; }
};

struct Minus
{
    template <typename A, typename B>
    __device__ static auto apply(const A a, const B b) { return a - b; }
};

struct Times
{
    template <typename A, typename B>
    __device__ static auto apply(const A a, const B b) { return a*b; }
};

struct Divide
{
    template <typename A, typename B>
    __device__ static auto apply(const A a, const B b) { return a/b; }
};

// `l Op r`, element by element. Nothing is computed when it is built: it holds
// its two operands - views and numbers, as cheap to copy as a pointer - and is
// evaluated in the one loop of the assignment it ends up in, so that
// `y = yTemp + dt*dy` is a single pass with no vector in between. A second
// statement is a second pass: nvcc does not fuse two loops, so two results
// computed from the same data are better written as one loop.
template <typename Op, typename L, typename R>
class ThreadVectorExpression
{
    L l_;
    R r_;

public:

    static constexpr label size =
        kThreadVectorOperandSize<L> != 0
      ? kThreadVectorOperandSize<L>
      : kThreadVectorOperandSize<R>;

    static_assert
    (
        kThreadVectorOperandSize<L> == 0
     || kThreadVectorOperandSize<R> == 0
     || kThreadVectorOperandSize<L> == kThreadVectorOperandSize<R>,
        "an expression of vectors of one size"
    );

    __device__ ThreadVectorExpression(const L& l, const R& r)
        : l_(l), r_(r)
    {}

    __device__ auto operator[](const label i) const
    {
        return Op::apply(element(l_, i), element(r_, i));
    }
};

// Either side a vector or an expression, the other one too or a number
template <typename L, typename R>
constexpr bool kAreThreadVectorOperands =
    (kIsThreadVectorOperand<L> && (kIsThreadVectorOperand<R> || kIsThreadVectorScalar<R>))
 || (kIsThreadVectorScalar<L> && kIsThreadVectorOperand<R>);

template <typename L, typename R, typename = std::enable_if_t<kAreThreadVectorOperands<L, R>>>
__device__ ThreadVectorExpression<Plus, L, R> operator+(const L& l, const R& r)
{
    return {l, r};
}

template <typename L, typename R, typename = std::enable_if_t<kAreThreadVectorOperands<L, R>>>
__device__ ThreadVectorExpression<Minus, L, R> operator-(const L& l, const R& r)
{
    return {l, r};
}

template <typename L, typename R, typename = std::enable_if_t<kAreThreadVectorOperands<L, R>>>
__device__ ThreadVectorExpression<Times, L, R> operator*(const L& l, const R& r)
{
    return {l, r};
}

template <typename L, typename R, typename = std::enable_if_t<kAreThreadVectorOperands<L, R>>>
__device__ ThreadVectorExpression<Divide, L, R> operator/(const L& l, const R& r)
{
    return {l, r};
}

// The calling thread's own vector of N elements in scratch space: element i is
// at INDEXVEC(i). N is a system unless said otherwise. A view, see
// DeviceBuffer.cuh: copied like a pointer, assigned like a reference.
template <typename T, label N = kSystemSize>
class ThreadVector
    :
    public DeviceBuffer<T>
{
public:

    using value_type = std::remove_const_t<T>;

    static constexpr label size = N;

    ThreadVector() = default;

    ThreadVector(const ThreadVector&) = default;

    __host__ __device__ explicit ThreadVector(T* data)
        : DeviceBuffer<T>(data)
    {}

    // a read only view of a vector that may be written
    template <typename U, typename = std::enable_if_t<std::is_same_v<const U, T>>>
    __host__ __device__ ThreadVector(const ThreadVector<U, N>& v)
        : DeviceBuffer<T>(v.data())
    {}

    __device__ T& operator[](const label i) const
    {
        return this->data_[INDEXVEC(i)];
    }

    // Assigning writes the elements, see DeviceBuffer.cuh. What is assigned
    // may be another vector, an expression of vectors or a number for all of
    // them.
    __device__ const ThreadVector& operator=(const ThreadVector& v) const
    {
        return update(v, [](T& x, const value_type e) { x = e; });
    }

    template <typename X, typename = std::enable_if_t<kIsThreadVectorOperand<X> || kIsThreadVectorScalar<X>>>
    __device__ const ThreadVector& operator=(const X& x) const
    {
        return update(x, [](T& y, const value_type e) { y = e; });
    }

    template <typename X, typename = std::enable_if_t<kIsThreadVectorOperand<X> || kIsThreadVectorScalar<X>>>
    __device__ const ThreadVector& operator+=(const X& x) const
    {
        return update(x, [](T& y, const value_type e) { y += e; });
    }

    template <typename X, typename = std::enable_if_t<kIsThreadVectorOperand<X> || kIsThreadVectorScalar<X>>>
    __device__ const ThreadVector& operator-=(const X& x) const
    {
        return update(x, [](T& y, const value_type e) { y -= e; });
    }

    template <typename X, typename = std::enable_if_t<kIsThreadVectorOperand<X> || kIsThreadVectorScalar<X>>>
    __device__ const ThreadVector& operator*=(const X& x) const
    {
        return update(x, [](T& y, const value_type e) { y *= e; });
    }

    template <typename X, typename = std::enable_if_t<kIsThreadVectorOperand<X> || kIsThreadVectorScalar<X>>>
    __device__ const ThreadVector& operator/=(const X& x) const
    {
        return update(x, [](T& y, const value_type e) { y /= e; });
    }

    // N elements per resident thread
    __host__ void allocate(const label scratchSize)
    {
        this->allocateElements(size_t(N) * scratchSize);
    }

private:

    // The one loop every assignment is: element i of x into element i of this
    template <typename X, typename Update>
    __device__ const ThreadVector& update(const X& x, const Update f) const
    {
        static_assert
        (
            kThreadVectorOperandSize<X> == 0 || kThreadVectorOperandSize<X> == N,
            "an assignment of a vector of the same size"
        );

        // Both copied before the loop, a view being a pointer: either may be a
        // member of the very object the loop writes through - a method's own
        // scratch - and nvcc would then read the pointer back out of it after
        // every element it stores
        const ThreadVector to = *this;
        const X from = x;

        #pragma unroll 4
        for (label i = 0; i < N; ++i)
        {
            f(to[i], element(from, i));
        }
        return *this;
    }
};

}

#endif
