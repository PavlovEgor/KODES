#ifndef KODES_THREAD_VECTOR_FUNCTIONS
#define KODES_THREAD_VECTOR_FUNCTIONS

#pragma once

#include "ThreadVector.cuh"

// Functions of vectors, named as OpenFOAM names them for its own: cmpt* works
// component by component and gives a vector - an expression, see
// ThreadVector.cuh - while mag, magSqr and the one argument cmptMax reduce a
// vector to a number. Not abs or max: a kodes::max would hide CUDA's
// max(double, double) from every unqualified call inside the namespace.
namespace kodes
{

// `Op x`, element by element; see ThreadVectorExpression
template <typename Op, typename X>
class ThreadVectorFunction
{
    X x_;

public:

    static constexpr label size = kThreadVectorOperandSize<X>;

    __device__ explicit ThreadVectorFunction(const X& x)
        : x_(x)
    {}

    __device__ auto operator[](const label i) const
    {
        return Op::apply(element(x_, i));
    }
};

template <typename Op, typename X>
constexpr bool kIsThreadVectorOperand<ThreadVectorFunction<Op, X>> = true;

template <typename Op, typename X>
constexpr label kThreadVectorOperandSize<ThreadVectorFunction<Op, X>> = kThreadVectorOperandSize<X>;

struct Mag
{
    template <typename A>
    __device__ static auto apply(const A a) { return fabs(a); }
};

struct Sqr
{
    template <typename A>
    __device__ static auto apply(const A a) { return a*a; }
};

struct Max
{
    template <typename A, typename B>
    __device__ static auto apply(const A a, const B b) { return max(a, b); }
};

template <typename X, typename = std::enable_if_t<kIsThreadVectorOperand<X>>>
__device__ ThreadVectorFunction<Mag, X> cmptMag(const X& x)
{
    return ThreadVectorFunction<Mag, X>(x);
}

template <typename X, typename = std::enable_if_t<kIsThreadVectorOperand<X>>>
__device__ ThreadVectorFunction<Sqr, X> cmptSqr(const X& x)
{
    return ThreadVectorFunction<Sqr, X>(x);
}

template <typename L, typename R, typename = std::enable_if_t<kAreThreadVectorOperands<L, R>>>
__device__ ThreadVectorExpression<Max, L, R> cmptMax(const L& l, const R& r)
{
    return {l, r};
}

// The reductions: one loop over x, in order, so that a sum adds up exactly as
// the loop it replaces did

template <typename X, typename = std::enable_if_t<kIsThreadVectorOperand<X>>>
__device__ auto magSqr(const X& x)
{
    decltype(Sqr::apply(element(x, 0))) sum = 0;

    #pragma unroll 4
    for (label i = 0; i < kThreadVectorOperandSize<X>; ++i)
    {
        sum += Sqr::apply(element(x, i));
    }
    return sum;
}

template <typename X, typename = std::enable_if_t<kIsThreadVectorOperand<X>>>
__device__ auto mag(const X& x)
{
    return sqrt(magSqr(x));
}

template <typename X, typename = std::enable_if_t<kIsThreadVectorOperand<X>>>
__device__ auto cmptMax(const X& x)
{
    auto result = element(x, 0);

    #pragma unroll 4
    for (label i = 1; i < kThreadVectorOperandSize<X>; ++i)
    {
        result = max(result, element(x, i));
    }
    return result;
}

}

#endif
